/* This file provides the implementation of the  vec< int64 , 2>  type
 * Only for including in emm_vec.hxx.
 * Don't include anywhere else.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified: 11-3-2015
 * modified 19-4-2012
 */


// load vector:
template<>  vec<int64_t, 2>::vec(const int64_t *v) {
  xmmi[0] = _mm_loadu_si128( reinterpret_cast<const __m128i *>( v ) );
}
template<> template<typename int_t> vec<int64_t, 2>::vec(const int64_t * v, int_t stride) {
  xmmi[0] = _mm_set_epi64x( *(v+stride), *v);
}
template<>  vec<int64_t, 2>::vec(const int64_t v) {
  xmmi[0] = _mm_set_epi64x( v, v);
}

//create as zero vector:
template <> vec<int64_t, 2> vec<int64_t, 2>::zero () {
  return vec<int64_t, 2>(_mm_setzero_si128());
}

// Store functions:
template <> void vec<int64_t, 2>::store(int64_t *v) {
  _mm_storeu_si128(  reinterpret_cast< __m128i *>  (v    ), xmmi[0]);
}
template <> void vec<int64_t, 2>::storea(int64_t *v) {
  _mm_store_si128(  reinterpret_cast< __m128i *>  (v    ), xmmi[0]);
}
template<> template<typename int_t>  void vec<int64_t, 2>::store(int64_t *v, int_t stride) {
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (v          ), xmm[0]);
  //_mm_storeh_epi64(  reinterpret_cast< __m128i *> (v + stride ), xmm[0]);
  __m128i tmp = _mm_unpackhi_epi64( xmmi[0],xmmi[0]);
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (v +  stride ), tmp);

}


// Operators, Specialized versions (int64_t precision, length 2):
template <> vec<int64_t, 2> vec<int64_t, 2>::operator* (const vec<int64_t, 2> &v) const {
  // Works for small values (magnitude < 2^32) , fast:
  //return vec<int64_t, 2>(  _mm_mul_epi32( xmmi[0], v.xmmi[0] ) );
  // works, but (relatively) slow:
  int64_t tmp1[2];
  int64_t tmp2[2];
  _mm_storeu_si128(  reinterpret_cast< __m128i *>  (&tmp1[0] ), xmmi[0]);
  _mm_storeu_si128(  reinterpret_cast< __m128i *>  (&tmp2[0] ), v.xmmi[0]);
  tmp1[0] = tmp1[0]*tmp2[0];
  tmp1[1] = tmp1[1]*tmp2[1];
  return vec<int64_t, 2>(  _mm_loadu_si128( reinterpret_cast<const __m128i *>( &tmp1[0] ) ) );
  // */
}
template <> vec<int64_t, 2> vec<int64_t, 2>::operator+ (const vec<int64_t, 2> &v) const {
  return vec<int64_t, 2>(  _mm_add_epi64( xmmi[0], v.xmmi[0] ) );
}
template <> vec<int64_t, 2> vec<int64_t, 2>::operator- (const vec<int64_t, 2> &v) const {
  return vec<int64_t, 2>(  _mm_sub_epi64( xmmi[0], v.xmmi[0] ) );
}
/*template <> vec<int64_t, 2> vec<int64_t, 2>::operator/ (const vec<int64_t, 2> &v) const {
  return vec<int64_t, 2>(  _mm_sub_epi64(xmmi[0], v.xmmi[0]) );
}*/
template <> inline void vec<int64_t, 2>::operator*= (const vec<int64_t, 2> &v) {
  //xmm[0]=  _mm_mul_epi32( xmmi[0], v.xmmi[0] );
  int64_t tmp1[2];
  int64_t tmp2[2];
  _mm_storeu_si128(  reinterpret_cast< __m128i *>  (&tmp1[0] ), xmmi[0]);
  _mm_storeu_si128(  reinterpret_cast< __m128i *>  (&tmp2[0] ), v.xmmi[0]);
  tmp1[0] = tmp1[0]*tmp2[0];
  tmp1[1] = tmp1[1]*tmp2[1];
  xmm[0]=  _mm_loadu_si128( reinterpret_cast<const __m128i *>( &tmp1[0] ) ) ;
}
template <> inline void vec<int64_t, 2>::operator+= (const vec<int64_t, 2> &v) {
  xmm[0]=  _mm_add_epi64(xmmi[0], v.xmmi[0] );
}
template <> inline void vec<int64_t, 2>::operator-= (const vec<int64_t, 2> &v) {
  xmm[0]=  _mm_sub_epi64(xmmi[0], v.xmmi[0] );
}
/*template <> inline void vec<int64_t, 2>::operator/= (const vec<int64_t, 2> &v) {
  xmm[0]=  _mm_mul_epi32(xmmi[0], v.xmmi[0] );
}*/

// Operators,  scalar versions (int64_t , length 2):
template <> vec<int64_t, 2> vec<int64_t, 2>::operator* (const int64_t v) const {
  //__m128i tmp = _mm_set_epi64x( v, v);
  //return vec<int64_t, 2>( _mm_mul_epi32(xmm[0], tmp));
  int64_t tmp1[2];
  _mm_storeu_si128(  reinterpret_cast< __m128i *>  (&tmp1[0] ), xmmi[0]);
  tmp1[0] = tmp1[0]*v;
  tmp1[1] = tmp1[1]*v;
  return vec<int64_t, 2>(  _mm_loadu_si128( reinterpret_cast<const __m128i *>( &tmp1[0] ) ) );
}
template <> vec<int64_t, 2> vec<int64_t, 2>::operator+ (const int64_t &v) const {
  __m128i tmp = _mm_set_epi64x( v, v);
  return vec<int64_t, 2>( _mm_add_epi64(xmm[0], tmp));
}
template <> vec<int64_t, 2> vec<int64_t, 2>::operator- (const int64_t &v) const {
  __m128i tmp = _mm_set_epi64x( v, v);
  return vec<int64_t, 2>( _mm_sub_epi64(xmm[0], tmp));
}
/*template <> vec<int64_t, 2> vec<int64_t, 2>::operator/ (const int64_t &v) const {
  __m128i tmp = _mm_set_epi64x( v, v);
  return vec<int64_t, 2>( _mm_sub_epi64(xmm[0], tmp));
}*/
template <> inline void vec<int64_t, 2>::operator*= (const int64_t &v) {
  //__m128i tmp = _mm_set_epi64x( v, v);
  //xmm[0] = _mm_mul_epi32(xmm[0], tmp);
  int64_t tmp1[2];
  _mm_storeu_si128(  reinterpret_cast< __m128i *>  (&tmp1[0] ), xmmi[0]);
  tmp1[0] = tmp1[0]*v;
  tmp1[1] = tmp1[1]*v;
  xmm[0] =  _mm_loadu_si128( reinterpret_cast<const __m128i *>( &tmp1[0] ) );
}
template <> inline void vec<int64_t, 2>::operator+= (const int64_t &v) {
  __m128i tmp = _mm_set_epi64x( v, v);
  xmm[0] = _mm_add_epi64(xmm[0], tmp);
}
template <> inline void vec<int64_t, 2>::operator-= (const int64_t &v) {
  __m128i tmp = _mm_set_epi64x( v, v);
  xmm[0] = _mm_sub_epi64(xmm[0], tmp);
}
/*template <> inline void vec<int64_t, 2>::operator/= (const int64_t &v) {
  __m128i tmp = _mm_set_epi64x( v, v);
  xmm[0] = _mm_div_epi32(xmm[0], tmp);
}*/
template <> inline vec<int64_t, 2> vec<int64_t, 2>::operator= (const int64_t &v) {
  xmm[0] = _mm_set_epi64x( v, v);
  return vec<int64_t, 2>( xmm[0] );
}

int64_t ALIGN16 negation64[2] = {0x8000000000000000,0x8000000000000000};
template <> vec<int64_t, 2> vec<int64_t, 2>::operator>> (const int shiftcount) const  {
  // shift right not supported by a single intrinsic for signed int64 values. (ARGH!!, why???)
  __m128i tmp = _mm_load_si128( (__m128i *) &negation64[0] );
  return vec<int64_t, 2>( _mm_sub_epi64( _mm_srli_epi64( _mm_add_epi64( xmmi[0], tmp), shiftcount), _mm_srli_epi64( tmp, shiftcount) ) );
}
template <> vec<double,2> vec<double,2>::signmask(){
    return vec<double,2>( (double *) &negation64[0] );
}
template <> vec<double,4> vec<double,4>::signmask(){
    __m128d tmp = _mm_load_pd( (double *) &negation64[0] );
    return vec<double,4>(tmp ,tmp   );
}
template <> vec<int64_t, 2> vec<int64_t, 2>::operator<< (const int shiftcount) const  {
  return vec<int64_t, 2>( _mm_slli_epi64(xmmi[0], shiftcount) );
}
template <> vec<int64_t, 2> vec<int64_t, 2>::operator| (const vec<int64_t, 2> &v) const {
  return vec<int64_t, 2>( _mm_or_si128(xmmi[0], v.xmmi[0]) );
}
template <> vec<int64_t, 2> vec<int64_t, 2>::operator& (const vec<int64_t, 2> &v) const {
  return vec<int64_t, 2>( _mm_and_si128(xmmi[0], v.xmmi[0]) );
}
#ifdef INCLUDE_SSE4
template <> vec<int64_t, 2> vec<int64_t, 2>::operator>(const vec<int64_t, 2> &v) const  {
  return vec<int64_t, 2>( _mm_cmpgt_epi64(xmmi[0], v.xmmi[0]) );
}
/* Somehow lt comparison of 64 bit integers is not supported (which idiot decided that?)
// obviously !((a>b) | (a==b)) is the same, but much slower.
template <> vec<int64_t, 2> vec<int64_t, 2>::operator<(const vec<int64_t, 2> &v) const  {
  return vec<int64_t, 2>( _mm_cmplt_epi64(xmmi[0], v.xmmi[0]) );
}*/
template <> vec<int64_t, 2> vec<int64_t, 2>::operator==(const vec<int64_t, 2> &v) const {
  return vec<int64_t, 2>( _mm_cmpeq_epi64(xmmi[0], v.xmmi[0]) );
}
#endif


#ifdef INCLUDE_SSE4
// other functions (min/max ..) (int64, length 2)
inline vec<int64_t, 2> max_bad(const vec<int64_t, 2> &a, const vec<int64_t, 2> &b){
    // _bad : using 32 bit maximum function
  return vec<int64_t,2>(_mm_max_epi32(a.xmmi[0], b.xmmi[0]));
}
inline vec<int64_t, 2> min_bad(const vec<int64_t, 2> &a, const vec<int64_t, 2> &b){
    // _bad : using 32 bit maximum function
  return vec<int64_t,2>(_mm_min_epi32(a.xmmi[0], b.xmmi[0]));
}
#endif
