#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>


using namespace std;
typedef vector<vector<float>> matrix;
//2D->1D
__device__  int idx(int i, int j, int nx) {
    return i + j * nx;
}


__global__ void compute_b_kernel(float* b, const float* u, const float* v, int nx, int ny, float dt, float dx, float dy, float rho) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int center = idx(i, j, nx);
        int E = idx(i + 1, j, nx); int W = idx(i - 1, j, nx);
        int N = idx(i, j + 1, nx); int S = idx(i, j - 1, nx);

        float dudx = (u[E] - u[W]) / (2 * dx);
        float dvdy = (v[N] - v[S]) / (2 * dy);
        float dudy = (u[N] - u[S]) / (2 * dy);
        float dvdx = (v[E] - v[W]) / (2 * dx);

        b[center] = rho * (1 / dt * (dudx + dvdy) - (dudx * dudx) - 2 * (dudy * dvdx) - (dvdy * dvdy));
    }
}


__global__ void pressure_poisson_kernel(float* p, const float* pn, const float* b, int nx, int ny, float dx, float dy) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int center = idx(i, j, nx);
        int E = idx(i + 1, j, nx); int W = idx(i - 1, j, nx);
        int N = idx(i, j + 1, nx); int S = idx(i, j - 1, nx);
        
        float dx2 = dx * dx;
        float dy2 = dy * dy;

        p[center] = (dy2 * (pn[E] + pn[W]) + dx2 * (pn[N] + pn[S]) - b[center] * dx2 * dy2) / (2 * (dx2 + dy2));
    }
}

__global__ void apply_pressure_bc_kernel(float* p, int nx, int ny) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    // 左右の境界
    if (i == 0 && j > 0 && j < ny - 1) p[idx(0, j, nx)] = p[idx(1, j, nx)];
    if (i == nx - 1 && j > 0 && j < ny-1) p[idx(nx - 1, j, nx)] = p[idx(nx - 2, j, nx)];

    // 上下の境界
    if (j == 0 && i > 0 && i < nx - 1) p[idx(i, 0, nx)] = p[idx(i, 1, nx)];
    if (j == ny - 1 && i > 0 && i < nx-1) p[idx(i, ny - 1, nx)] = 0.0f;
}

// 速度場を更新するカーネル
__global__ void update_velocity_kernel(float* u, float* v, const float* un, const float* vn, const float* p, int nx, int ny, float dt, float dx, float dy, float rho, float nu) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int center = idx(i, j, nx);
        int E = idx(i + 1, j, nx); int W = idx(i - 1, j, nx);
        int N = idx(i, j + 1, nx); int S = idx(i, j - 1, nx);

        u[center] = un[center] - un[center] * dt / dx * (un[center] - un[W])
                               - un[center] * dt / dy * (un[center] - un[S])
                               - dt / (2 * rho * dx) * (p[E] - p[W])
                               + nu * dt / (dx * dx) * (un[E] - 2 * un[center] + un[W])
                               + nu * dt / (dy * dy) * (un[N] - 2 * un[center] + un[S]);

        v[center] = vn[center] - vn[center] * dt / dx * (vn[center] - vn[W])
                               - vn[center] * dt / dy * (vn[center] - vn[S])
                               - dt / (2 * rho * dy) * (p[N] - p[S])
                               + nu * dt / (dx * dx) * (vn[E] - 2 * vn[center] + vn[W])
                               + nu * dt / (dy * dy) * (vn[N] - 2 * vn[center] + vn[S]);
    }
}

__global__ void apply_velocity_bc_kernel(float* u, float* v, int nx, int ny) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < nx && j < ny) {
        if (i == 0 || i == nx - 1 || j == 0) {
            u[idx(i, j, nx)] = 0.0f;
            v[idx(i, j, nx)] = 0.0f;
        }
        if (j == ny - 1) {
            u[idx(i, j, nx)] = 1.0f;
            v[idx(i, j, nx)] = 0.0f;
        }
    }
}

// void write_to_file(const std::string& filename, const float* h_data, int nx, int ny) {
//     static std::map<std::string, std::ofstream> files;
//     if (files.find(filename) == files.end()) {
//         files[filename].open(filename);
//     }
//     for (int j = 0; j < ny; j++) {
//         for (int i = 0; i < nx; i++) {
//             files[filename] << h_data[idx(i, j, nx)] << " ";
//         }
//     }
//     files[filename] << "\n";
// }

int main() {
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;
  size_t size = nx * ny * sizeof(float);

  float *h_u = (float*)malloc(size);
  float *h_v = (float*)malloc(size);
  float *h_p = (float*)malloc(size);

  float *d_u, *d_v, *d_p, *d_b, *d_un, *d_vn, *d_pn;
  cudaMalloc((void**)&d_u, size);
  cudaMalloc((void**)&d_v, size);
  cudaMalloc((void**)&d_p, size);
  cudaMalloc((void**)&d_b, size);
  cudaMalloc((void**)&d_un, size);
  cudaMalloc((void**)&d_vn, size);
  cudaMalloc((void**)&d_pn, size);

  cudaMemset(d_u, 0, size);
  cudaMemset(d_v, 0, size);
  cudaMemset(d_p, 0, size);
  cudaMemset(d_b, 0, size);
  
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                  (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");
  for (int n = 0; n < nt; n++) {
      // 1. bの計算
      compute_b_kernel<<<numBlocks, threadsPerBlock>>>(d_b, d_u, d_v, nx, ny, dt, dx, dy, rho);
      cudaGetLastError();

      // 2. 圧力の反復計算
      for (int it = 0; it < nit; it++) {
          cudaMemcpy(d_pn, d_p, size, cudaMemcpyDeviceToDevice);
          pressure_poisson_kernel<<<numBlocks, threadsPerBlock>>>(d_p, d_pn, d_b, nx, ny, dx, dy);
          cudaGetLastError();
          apply_pressure_bc_kernel<<<numBlocks, threadsPerBlock>>>(d_p, nx, ny);
          cudaGetLastError();
      }

      // 3. 速度のコピー
      cudaMemcpy(d_un, d_u, size, cudaMemcpyDeviceToDevice);
      cudaMemcpy(d_vn, d_v, size, cudaMemcpyDeviceToDevice);

      // 4. 速度の更新
      update_velocity_kernel<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_un, d_vn, d_p, nx, ny, dt, dx, dy, rho, nu);
      cudaGetLastError();
      
      // 5. 速度の境界条件適用
      apply_velocity_bc_kernel<<<numBlocks, threadsPerBlock>>>(d_u, d_v, nx, ny);
      cudaGetLastError();


      if (n % 10 == 0) {
        cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_v, d_v, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p, d_p, size, cudaMemcpyDeviceToHost);
        for (int j=0; j<ny; j++)
            for (int i=0; i<nx; i++)
            ufile << h_u[i + j * nx] << " ";
        ufile << "\n";
        for (int j=0; j<ny; j++)
            for (int i=0; i<nx; i++)
            vfile << h_v[i + j * nx] << " ";
        vfile << "\n";
        for (int j=0; j<ny; j++)
            for (int i=0; i<nx; i++)
            pfile << h_p[i + j * nx] << " ";
        pfile << "\n";
        }
  }

  // メモリの解放
  free(h_u);
  free(h_v);
  free(h_p);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_p);
  cudaFree(d_b);
  cudaFree(d_un);
  cudaFree(d_vn);
  cudaFree(d_pn);

  return 0;

}
