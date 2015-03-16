/* This file provides the implementation of the  vec< int64 , 3>  type
 * Only for including in emm_vec.hxx.
 * Don't include anywhere else.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 12-3-2012
 */


// load vector:
template<>  vec<uint64_t, 3>::vec(const uint64_t *v) {
  xmmi[0] = _mm_loadu_si128( reinterpret_cast<const __m128i *>( v     ) );
  xmmi[1] = _mm_set_epi64x( 0, *(v+2));
}
template<> template<typename int_t> vec<uint64_t, 3>::vec(const uint64_t * v, int_t stride) {
  xmmi[0] = _mm_set_epi64x( *(v + 1*stride), *(v + 0*stride) );
  xmmi[1] = _mm_set_epi64x( 0              , *(v + 2*stride) );
}
template<>  vec<uint64_t, 3>::vec(const uint64_t v) {
  xmmi[0] =  _mm_set_epi64x( v, v);
  xmmi[1] = xmmi[0];
}

//create as zero vector:
template <> vec<uint64_t, 3> vec<uint64_t, 3>::zero () {
  return vec<uint64_t, 3>(_mm_setzero_si128(), _mm_setzero_si128());
}

// Store functions:
template <> void vec<uint64_t, 3>::store(uint64_t *v) {
  _mm_storeu_si128( reinterpret_cast< __m128i *>(v    ), xmmi[0]);
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (v +2 ), xmm[1]);
}
template <> void vec<uint64_t, 3>::storea(uint64_t *v) {
  _mm_store_si128( reinterpret_cast< __m128i *>(v    ), xmmi[0]);
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (v +2 ), xmm[1]);
}
template<> template<typename int_t>  void vec<uint64_t, 3>::store(uint64_t *v, int_t stride) {
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (v           ), xmm[0]);
  //_mm_storeh_epi64(  reinterpret_cast< __m128i *> (v +  stride ), xmm[0]);
  __m128i tmp0 = _mm_unpackhi_epi64( xmmi[0],xmmi[0]);
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (v +  stride ), tmp0);
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (v +2*stride ), xmm[1]);
}


// Operators, Specialized versions (uint64_t precision, length 3):
template <> vec<uint64_t, 3> vec<uint64_t, 3>::operator* (const vec<uint64_t, 3> &v) const {
  //return vec<uint64_t, 3>(  _mm_mul_epi32( xmmi[0], v.xmmi[0] ) , _mm_mul_epi32( xmmi[1], v.xmmi[1] ));
  uint64_t tmp1[3];
  uint64_t tmp2[3];
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp1[0] ),   xmm[0]);
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (&tmp1[2] ),   xmm[1]);
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp2[0] ), v.xmm[0]);
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (&tmp2[2] ), v.xmm[1]);
  tmp1[0] = tmp1[0] * tmp2[0];
  tmp1[1] = tmp1[1] * tmp2[1];
  tmp1[2] = tmp1[2] * tmp2[2];
  return vec<uint64_t, 3>( _mm_loadu_si128( reinterpret_cast<const __m128i *>( &tmp1[0] ) ), 
                           _mm_loadl_epi64( reinterpret_cast<const __m128i *>( &tmp1[2] ) ) );
}
template <> vec<uint64_t, 3> vec<uint64_t, 3>::operator+ (const vec<uint64_t, 3> &v) const {
  return vec<uint64_t, 3>(  _mm_add_epi64(xmmi[0], v.xmmi[0]) , _mm_add_epi64(xmmi[1], v.xmmi[1]) );
}
template <> vec<uint64_t, 3> vec<uint64_t, 3>::operator- (const vec<uint64_t, 3> &v) const {
  return vec<uint64_t, 3>(  _mm_sub_epi64(xmmi[0], v.xmmi[0]), _mm_sub_epi64(xmmi[1], v.xmmi[1]) );
}
/*template <> vec<uint64_t, 3> vec<uint64_t, 3>::operator/ (const vec<uint64_t, 3> &v) const {
  return vec<uint64_t, 3>(  _mm_sub_epi64(xmmi[0], v.xmmi[0]) );
}*/
template <> inline void vec<uint64_t, 3>::operator*= (const vec<uint64_t, 3> &v) {
  //xmm[0]=  _mm_mul_epi32( xmmi[0], v.xmmi[0] );
  //xmm[1]=  _mm_mul_epi32( xmmi[1], v.xmmi[1] );
  uint64_t tmp1[3];
  uint64_t tmp2[3];
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp1[0] ),   xmm[0]);
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (&tmp1[2] ),   xmm[1]);
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp2[0] ), v.xmm[0]);
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (&tmp2[2] ), v.xmm[1]);
  tmp1[0] = tmp1[0] * tmp2[0];
  tmp1[1] = tmp1[1] * tmp2[1];
  tmp1[2] = tmp1[2] * tmp2[2];
  xmm[0] = _mm_loadu_si128( reinterpret_cast<const __m128i *>( &tmp1[0] ) ); 
  xmm[1] = _mm_loadl_epi64( reinterpret_cast<const __m128i *>( &tmp1[2] ) ) ;
}
template <> inline void vec<uint64_t, 3>::operator+= (const vec<uint64_t, 3> &v) {
  xmm[0]=  _mm_add_epi64(xmmi[0], v.xmmi[0] );
  xmm[1]=  _mm_add_epi64(xmmi[1], v.xmmi[1] );
}
template <> inline void vec<uint64_t, 3>::operator-= (const vec<uint64_t, 3> &v) {
  xmm[0]=  _mm_sub_epi64(xmmi[0], v.xmmi[0] );
  xmm[1]=  _mm_sub_epi64(xmmi[1], v.xmmi[1] );
}
/*template <> inline void vec<uint64_t, 3>::operator/= (const vec<uint64_t, 3> &v) {
  xmm[0]=  _mm_mul_epi32(xmmi[0], v.xmmi[0] );
  xmm[0]=  _mm_mul_epi32(xmmi[0], v.xmmi[0] );
}*/

// Operators,  scalar versions (uint64_t , length 3):
template <> vec<uint64_t, 3> vec<uint64_t, 3>::operator* (const uint64_t v) const {
  //__m128i tmp = _mm_set_epi64x( v, v);
  //return vec<uint64_t, 3>( _mm_mul_epi32(xmm[0], tmp), _mm_mul_epi32(xmm[1], tmp));
  uint64_t tmp1[3];
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp1[0] ), xmm[0]);
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (&tmp1[2] ), xmm[1]);
  tmp1[0] = tmp1[0]*v;
  tmp1[1] = tmp1[1]*v;
  tmp1[2] = tmp1[2]*v;
  return vec<uint64_t, 3>( _mm_loadu_si128( reinterpret_cast<const __m128i *>( &tmp1[0] ) ), 
                           _mm_loadl_epi64( reinterpret_cast<const __m128i *>( &tmp1[2] ) ) );
}
template <> vec<uint64_t, 3> vec<uint64_t, 3>::operator+ (const uint64_t &v) const {
  __m128i tmp = _mm_set_epi64x( v, v);
  return vec<uint64_t, 3>( _mm_add_epi64(xmm[0], tmp), _mm_add_epi64(xmm[1], tmp));
}
template <> vec<uint64_t, 3> vec<uint64_t, 3>::operator- (const uint64_t &v) const {
  __m128i tmp = _mm_set_epi64x( v, v);
  return vec<uint64_t, 3>( _mm_sub_epi64(xmm[0], tmp), _mm_sub_epi64(xmm[1], tmp));
}
/*template <> vec<uint64_t, 3> vec<uint64_t, 3>::operator/ (const uint64_t &v) const {
  __m128i tmp = _mm_set_epi64x( v, v);
  return vec<uint64_t, 3>( _mm_sub_epi64(xmm[0], tmp));
}*/
template <> inline void vec<uint64_t, 3>::operator*= (const uint64_t &v) {
  /*__m128i tmp = _mm_set_epi64x( v, v);
  xmm[0] = _mm_mul_epi32(xmm[0], tmp);
  xmm[1] = _mm_mul_epi32(xmm[1], tmp);*/
  uint64_t tmp1[3];
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp1[0] ), xmm[0]);
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (&tmp1[2] ), xmm[1]);
  tmp1[0] = tmp1[0]*v;
  tmp1[1] = tmp1[1]*v;
  tmp1[2] = tmp1[2]*v;
  xmm[0] = _mm_loadu_si128( reinterpret_cast<const __m128i *>( &tmp1[0] ) ); 
  xmm[1] = _mm_loadl_epi64( reinterpret_cast<const __m128i *>( &tmp1[2] ) );

}
template <> inline void vec<uint64_t, 3>::operator+= (const uint64_t &v) {
  __m128i tmp = _mm_set_epi64x( v, v);
  xmm[0] = _mm_add_epi64(xmm[0], tmp);
  xmm[1] = _mm_add_epi64(xmm[1], tmp);
}
template <> inline void vec<uint64_t, 3>::operator-= (const uint64_t &v) {
  __m128i tmp = _mm_set_epi64x( v, v);
  xmm[0] = _mm_sub_epi64(xmm[0], tmp);
  xmm[1] = _mm_sub_epi64(xmm[1], tmp);
}
/*template <> inline void vec<uint64_t, 3>::operator/= (const uint64_t &v) {
  __m128i tmp = _mm_set_epi64x( v, v);
  xmm[0] = _mm_div_epi32(xmm[0], tmp);
  xmm[0] = _mm_div_epi32(xmm[0], tmp);
}*/
template <> inline vec<uint64_t, 3> vec<uint64_t, 3>::operator= (const uint64_t &v) {
  xmm[0] = _mm_set_epi64x( v, v);
  xmm[1] = xmm[0];
  return vec<uint64_t, 3>( xmm[0], xmm[0] );
}



template <> vec<uint64_t, 3> vec<uint64_t, 3>::operator>> (const int shiftcount) const  {
  return vec<uint64_t, 3>(  _mm_srli_epi64( xmmi[0], shiftcount) ,
                            _mm_srli_epi64( xmmi[1], shiftcount) );
}

template <> vec<uint64_t, 3> vec<uint64_t, 3>::operator<< (const int shiftcount) const  {
  return vec<uint64_t, 3>( _mm_slli_epi64(xmmi[0], shiftcount) , _mm_slli_epi64(xmmi[1], shiftcount) );
}
template <> vec<uint64_t, 3> vec<uint64_t, 3>::operator| (const vec<uint64_t, 3> &v) const {
  return vec<uint64_t, 3>( _mm_or_si128(xmmi[0], v.xmmi[0]), _mm_or_si128(xmmi[1], v.xmmi[1]) );
}
template <> vec<uint64_t, 3> vec<uint64_t, 3>::operator& (const vec<uint64_t, 3> &v) const {
  return vec<uint64_t, 3>( _mm_and_si128(xmmi[0], v.xmmi[0]), _mm_and_si128(xmmi[1], v.xmmi[1]) );
}
#ifdef INCLUDE_SSE4
template <> vec<uint64_t, 3> vec<uint64_t, 3>::operator>(const vec<uint64_t, 3> &v) const  {
  return vec<uint64_t, 3>( _mm_cmpgt_epi64(xmmi[0], v.xmmi[0]), _mm_cmpgt_epi64(xmmi[1], v.xmmi[1]) );
}
/* Somehow lt comparison of 64 bit integers is not supported
// obviously !((a>b) | (a==b)) is the same, but much slower.
template <> vec<uint64_t, 3> vec<uint64_t, 3>::operator<(const vec<uint64_t, 3> &v) const  {
  return vec<uint64_t, 3>( _mm_cmplt_epi64(xmmi[0], v.xmmi[0]) );
}*/
template <> vec<uint64_t, 3> vec<uint64_t, 3>::operator==(const vec<uint64_t, 3> &v) const {
  return vec<uint64_t, 3>( _mm_cmpeq_epi64(xmmi[0], v.xmmi[0]) ,  _mm_cmpeq_epi64(xmmi[1], v.xmmi[1]) );
}
#endif


#ifdef INCLUDE_SSE4
// other functions (min/max ..) (int64, length 3)
inline vec<uint64_t, 3> max_bad(const vec<uint64_t, 3> &a, const vec<uint64_t, 3> &b){
    // _bad : using 32 bit maximum function
  return vec<uint64_t,3>(_mm_max_epi32(a.xmmi[0], b.xmmi[0]), _mm_max_epi32(a.xmmi[1], b.xmmi[1]));
}
inline vec<uint64_t, 3> min_bad(const vec<uint64_t, 3> &a, const vec<uint64_t, 3> &b){
    // _bad : using 32 bit maximum function
  return vec<uint64_t,3>(_mm_min_epi32(a.xmmi[0], b.xmmi[0]), _mm_min_epi32(a.xmmi[1], b.xmmi[1]));
}
#endif
