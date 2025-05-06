import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import pywt
from sklearn.cluster import KMeans
from scipy.ndimage import zoom
from flask import Flask, render_template_string, request, send_file
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

HTML_TEMPLATE = """
<!doctype html>
<title>Chip Topology Analyzer</title>
<h2>Upload a .xyz file for Wavelet Analysis</h2>
<form method=post enctype=multipart/form-data action="/analyze">
  <input type=file name=file><br><br>
  <input type=submit value=Upload>
</form>
{% if image_paths %}
  <h3>Results:</h3>
  {% for path in image_paths %}
    <p><img src="{{ path }}" style="width: 45%; margin: 10px;"></p>
  {% endfor %}
{% endif %}
"""

# 3D Surface Plot
def plot_3d_surface(x, y, z):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, z, cmap='viridis', linewidth=0, antialiased=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Height)')
    ax.set_title('3D Surface Plot of Chip Topology')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_FOLDER, '3D_Surface_Plot.png'))
    plt.close()

# Z Histogram Plot
def plot_z_histogram(z):
    z_hist, bins = np.histogram(z, bins=40)
    plt.figure(figsize=(6, 5))
    plt.bar(bins[:-1], z_hist, width=np.diff(bins), edgecolor='black')
    plt.title('Z Histogram')
    plt.xlabel('Z value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_FOLDER, 'Z_Histogram.png'))
    plt.close()

# 2D Histogram Plot
def plot_2d_histogram(x, y):
    plt.figure(figsize=(10, 6))
    plt.hist2d(x, y, bins=50, cmap='Blues')
    plt.colorbar(label='Counts')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Histogram of X and Y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_FOLDER, '2D_Histogram.png'))
    plt.close()

# 3D Histogram Plot
def plot_3d_histogram(x, y):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(x, y, bins=50)
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.flatten()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Counts')
    plt.title('3D Histogram')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_FOLDER, '3D_Histogram.png'))
    plt.close()

def process_3d_wavelet(data):
    # Perform 3D wavelet transform
    coeffs3 = pywt.dwtn(data, 'db2')  
    
    # Print the keys of the coefficients to see the structure
    print(coeffs3.keys())  
    
    # Extract each frequency band using appropriate keys (adjusted based on the output)
    LL = coeffs3['aa']  # Low-Low frequency approximation
    LH = coeffs3['ad']  # High frequency in X direction
    HL = coeffs3['da']  # High frequency in Y direction
    HH = coeffs3['dd']  # High frequency in XY direction

    return LL, LH, HL, HH

# Detect Defects Using Clustering
def detect_defects(x, y, z):
    data = np.vstack([x, y, z]).T
    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(data)
    defect_mask = labels == 1
    return defect_mask

def save_img(data, name, cmap='viridis'):
    path = os.path.join(RESULT_FOLDER, f'{name}.png')
    plt.figure(figsize=(6, 5))
    plt.imshow(data, cmap=cmap)
    plt.title(name)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return '/' + path

def process_xyz(filepath):
    data = np.loadtxt(filepath)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    # Interpolate to grid (256x256 for consistency)
    grid_x, grid_y = np.mgrid[min(x):max(x):256j, min(y):max(y):256j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # Fill NaNs with median
    grid_z = np.nan_to_num(grid_z, nan=np.nanmedian(grid_z))

    # Perform 3D Wavelet Transform
    LL_3d, LH_3d, HL_3d, HH_3d = process_3d_wavelet(grid_z)

    # Rescale LL to match grid_z dimensions
    LL_rescaled = zoom(LL_3d, (grid_z.shape[0] / LL_3d.shape[0], grid_z.shape[1] / LL_3d.shape[1]))

    # Save Results
    paths = []
    paths.append(save_img(grid_z, 'Original_Heightmap'))
    paths.append(save_img(LL_rescaled, 'LL_Approximation'))
    paths.append(save_img(LH_3d, 'LH_HighFreq_X'))
    paths.append(save_img(HL_3d, 'HL_HighFreq_Y'))
    paths.append(save_img(HH_3d, 'HH_HighFreq_XY', cmap='seismic'))
    paths.append(save_img(grid_z - LL_rescaled, 'Defect_Mask_Approx'))

    # Plot histograms
    plot_z_histogram(grid_z)
    plot_2d_histogram(x, y)
    plot_3d_histogram(x, y)

    paths.append('/static/results/Z_Histogram.png')
    paths.append('/static/results/2D_Histogram.png')
    paths.append('/static/results/3D_Histogram.png')

    # 3D Surface Plot
    plot_3d_surface(x, y, z)
    paths.append('/static/results/3D_Surface_Plot.png')

    return paths

@app.route('/', methods=['GET'])
def upload_form():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    if not file or not file.filename.endswith('.xyz'):
        return 'Please upload a valid .xyz file'

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    image_paths = process_xyz(filepath)
    return render_template_string(HTML_TEMPLATE, image_paths=image_paths)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
