#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  const __mmask16 k_n_mask = 0xFF; 
  

  for(int i=0; i<N; i++) {
    const __mmask16 mask = 1 << i;
    __m512 zerovec = _mm512_setzero_ps();
    __m512 xi_vec = _mm512_set1_ps(x[i]);
    __m512 yi_vec = _mm512_set1_ps(y[i]);
    __m512 x_j = _mm512_load_ps(x);
    __m512 y_j = _mm512_load_ps(y);
    __m512 m_j = _mm512_load_ps(m);
    xi_vec=_mm512_mask_blend_ps(mask,xi_vec,zerovec);
    yi_vec=_mm512_mask_blend_ps(mask,yi_vec,zerovec);
    __m512 rx_vec = _mm512_sub_ps(xi_vec, x_j);
    __m512 ry_vec = _mm512_sub_ps(yi_vec, y_j);
    __m512 rx2_vec = _mm512_mul_ps(rx_vec, rx_vec);
    __m512 ry2_vec = _mm512_mul_ps(ry_vec, ry_vec);
    
    __m512 r2_vec = _mm512_add_ps(rx2_vec, ry2_vec);

    __m512 inv_r_vec = _mm512_rsqrt14_ps(r2_vec);
    
    __m512 inv_r3_vec = _mm512_mul_ps(inv_r_vec, inv_r_vec);
           inv_r3_vec = _mm512_mul_ps(inv_r_vec, inv_r3_vec);

    __m512 fx_vec = _mm512_mul_ps(rx_vec, m_j);
           fx_vec = _mm512_mul_ps(fx_vec, inv_r3_vec);
    
    __m512 fy_vec = _mm512_mul_ps(ry_vec, m_j);
           fy_vec = _mm512_mul_ps(fy_vec, inv_r3_vec);
    fx[i] -= _mm512_reduce_add_ps(fx_vec);
    fy[i] -= _mm512_reduce_add_ps(fy_vec);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
