/* This file provides the implementation of the  vec< double , 2>  type
 * Only for including in emm_vec.hxx.
 * Don't include anywhere else.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 12-3-2015
 */


// Type conversion constructors; convert between vectors of length 2 :

/*// double to int64
#ifdef INCLUDE_SSE4
template<> template<> inline vec<int64_t, 2>::vec(const vec<double, 2> &v) {
  xmmi[0] = _mm_cvtpd_epi32(v.xmmd[0]);
  xmmi[0] = _mm_cvtepi32_epi64(xmmi[0]);
}
#endif*/

// signed int32_t to unsigned int32_t:
template<> template<> inline vec<uint32_t, 3>::vec(const vec<int32_t, 3> &v) {
  xmm[0] = v.xmm[0];
}
// unsigned int32_t to signed int32_t:
template<> template<> inline vec<int32_t, 3>::vec(const vec<uint32_t, 3> &v) {
  xmm[0] = v.xmm[0];
}