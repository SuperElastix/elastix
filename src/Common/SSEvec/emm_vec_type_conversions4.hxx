/* This file provides the implementation of the  vec< double , 2>  type
 * Only for including in emm_vec.hxx.
 * Don't include anywhere else.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 12-3-2015
 */


// Type conversion constructors; convert between vectors of length 4 :

/*// double to int64
#ifdef INCLUDE_SSE4
template<> template<> inline vec<int64_t, 2>::vec(const vec<double, 4> &v) {
  xmmi[0] = _mm_cvtpd_epi32(v.xmmd[0]);
  xmmi[0] = _mm_cvtepi32_epi64(xmmi[0]);
}
#endif*/

// signed int32_t to unsigned int32_t:
template<> template<> inline vec<uint32_t, 4>::vec(const vec<int32_t, 4> &v) {
  xmm[0] = v.xmm[0];
}
// unsigned int32_t to signed int32_t:
template<> template<> inline vec<int32_t, 4>::vec(const vec<uint32_t, 4> &v) {
  xmm[0] = v.xmm[0];
}
// unsigned int32_t to unsigned int64_t:
template<> template<> inline vec<uint64_t, 4>::vec(const vec<uint32_t, 4> &v) {
  __m128i tmp = _mm_setzero_si128 ();
  xmm[0] = _mm_unpacklo_epi32( v.xmm[0] ,tmp);
  xmm[1] = _mm_unpackhi_epi32( v.xmm[0] ,tmp);
}
// unsigned int32_t to signed int64_t:
template<> template<> inline vec<int64_t, 4>::vec(const vec<uint32_t, 4> &v) {
  __m128i tmp = _mm_setzero_si128 ();
  xmm[0] = _mm_unpacklo_epi32( v.xmm[0] ,tmp);
  xmm[1] = _mm_unpackhi_epi32( v.xmm[0] ,tmp);
}
// int32_t to int64_t:
template<> template<> inline vec<int64_t, 4>::vec(const vec<int32_t, 4> &v) {
  __m128i tmp = _mm_srai_epi32( v.xmm[0] , 31);
  xmm[0] = _mm_unpacklo_epi32( v.xmm[0] ,tmp);
  xmm[1] = _mm_unpackhi_epi32( v.xmm[0] ,tmp);
}
// int32_t to unsigned int64_t:
template<> template<> inline vec<uint64_t, 4>::vec(const vec<int32_t, 4> &v) {
  __m128i tmp = _mm_srai_epi32( v.xmm[0] , 31);
  xmm[0] = _mm_unpacklo_epi32( v.xmm[0] ,tmp);
  xmm[1] = _mm_unpackhi_epi32( v.xmm[0] ,tmp);
}

// unsigned int64_t to unsigned int32_t:
template<> template<> inline vec<uint32_t, 4>::vec(const vec<uint64_t, 4> &v) {
  __m128i tmp0 = _mm_shuffle_epi32(  v.xmm[0] , _MM_SHUFFLE(3, 1, 2, 0));
  __m128i tmp1 = _mm_shuffle_epi32(  v.xmm[1] , _MM_SHUFFLE(3, 1, 2, 0));
  xmm[0] = _mm_unpacklo_epi64( tmp0, tmp1 );
}
// signed int64_t to unsigned int32_t:
template<> template<> inline vec<uint32_t, 4>::vec(const vec<int64_t, 4> &v) {
  __m128i tmp0 = _mm_shuffle_epi32(  v.xmm[0] , _MM_SHUFFLE(3, 1, 2, 0));
  __m128i tmp1 = _mm_shuffle_epi32(  v.xmm[1] , _MM_SHUFFLE(3, 1, 2, 0));
  xmm[0] = _mm_unpacklo_epi64( tmp0, tmp1 );
}

// unsigned int64_t to int32_t:
template<> template<> inline vec<int32_t, 4>::vec(const vec<uint64_t, 4> &v) {
  __m128i tmp0 = _mm_shuffle_epi32(  v.xmm[0] , _MM_SHUFFLE(3, 1, 2, 0));
  __m128i tmp1 = _mm_shuffle_epi32(  v.xmm[1] , _MM_SHUFFLE(3, 1, 2, 0));
  xmm[0] = _mm_unpacklo_epi64( tmp0, tmp1 );
}
// signed int64_t to int32_t:
template<> template<> inline vec<int32_t, 4>::vec(const vec<int64_t, 4> &v) {
  __m128i tmp0 = _mm_shuffle_epi32(  v.xmm[0] , _MM_SHUFFLE(3, 1, 2, 0));
  __m128i tmp1 = _mm_shuffle_epi32(  v.xmm[1] , _MM_SHUFFLE(3, 1, 2, 0));
  xmm[0] = _mm_unpacklo_epi64( tmp0, tmp1 );
}

// signed int64_t to unsigned int64_t:
template<> template<> inline vec<uint64_t, 4>::vec(const vec<int64_t, 4> &v) {
  xmm[0] = v.xmm[0];
  xmm[1] = v.xmm[1];
}
// unsigned int64_t to signed int64_t:
template<> template<> inline vec<int64_t, 4>::vec(const vec<uint64_t, 4> &v) {
  xmm[0] = v.xmm[0];
  xmm[1] = v.xmm[1];
}