/* This file provides the implementation of the  vec< int64 , 4>  type
 * Only for including in emm_vec.hxx.
 * Don't include anywhere else.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 12-3-2012
 */


// load vector:
template<>  vec<int64_t, 4>::vec(const int64_t *v) {
  xmmi[0] = _mm_loadu_si128( reinterpret_cast<const __m128i *>( v     ) );
  xmmi[1] = _mm_loadu_si128( reinterpret_cast<const __m128i *>( v + 2 ) );
}
template<> template<typename int_t> vec<int64_t, 4>::vec(const int64_t * v, int_t stride) {
  xmmi[0] = _mm_set_epi64x( *(v +   stride) , *v           );
  xmmi[1] = _mm_set_epi64x( *(v + 3*stride) , *(v+2*stride));
}
template<>  vec<int64_t, 4>::vec(const int64_t v) {
  xmmi[0] =  _mm_set_epi64x( v, v);
  xmmi[1] = xmmi[0];
}
template<> vec<int64_t, 4>  vec<int64_t, 4>::loada( const int64_t * v ) {
  //return vec( v ); // can't do aligned more efficient. 
  return vec<int64_t,4>( _mm_load_si128( reinterpret_cast<const __m128i *>( v ) ) , _mm_load_si128( reinterpret_cast<const __m128i *>( v +2 ) )  );
}

//create as zero vector:
template <> vec<int64_t, 4> vec<int64_t, 4>::zero () {
  return vec<int64_t, 4>(_mm_setzero_si128(), _mm_setzero_si128());
}

// Store functions:
template <> void vec<int64_t, 4>::store(int64_t *v) {
  _mm_storeu_si128( reinterpret_cast< __m128i *>(v    ), xmm[0]);
  _mm_storeu_si128( reinterpret_cast< __m128i *>(v+2  ), xmm[1]);
}
template <> void vec<int64_t, 4>::storea(int64_t *v) {
  _mm_store_si128( reinterpret_cast< __m128i *>(v    ), xmm[0]);
  _mm_store_si128( reinterpret_cast< __m128i *>(v+2  ), xmm[1]);
}
template<> template<typename int_t>  void vec<int64_t, 4>::store(int64_t *v, int_t stride) {
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (v           ), xmm[0]);
//  _mm_storeh_epi64(  reinterpret_cast< __m128i *> (v +  stride ), xmm[0]);
  __m128i tmp0 = _mm_unpackhi_epi64( xmmi[0],xmmi[0]);
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (v +  stride ), tmp0);
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (v +2*stride ), xmm[1]);
//  _mm_storeh_epi64(  reinterpret_cast< __m128i *> (v +3*stride ), xmm[1]);
  __m128i tmp1 = _mm_unpackhi_epi64( xmm[1],xmm[1]);
  _mm_storel_epi64(  reinterpret_cast< __m128i *> (v +3*stride ), tmp1);

}


// Operators, Specialized versions (int64_t precision, length 4):
template <> vec<int64_t, 4> vec<int64_t, 4>::operator* (const vec<int64_t, 4> &v) const {
  //return vec<int64_t, 4>(  _mm_mul_epi32( xmmi[0], v.xmmi[0] ) , _mm_mul_epi32( xmmi[1], v.xmmi[1] ));
  int64_t tmp1[4];
  int64_t tmp2[4];
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp1[0] ),   xmm[0]);
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp1[2] ),   xmm[1]);
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp2[0] ), v.xmm[0]);
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp2[2] ), v.xmm[1]);
  tmp1[0] = tmp1[0] * tmp2[0];
  tmp1[1] = tmp1[1] * tmp2[1];
  tmp1[2] = tmp1[2] * tmp2[2];
  tmp1[3] = tmp1[3] * tmp2[3];
  return vec<int64_t, 4>( _mm_loadu_si128( reinterpret_cast<const __m128i *>( &tmp1[0] ) ), 
                          _mm_loadu_si128( reinterpret_cast<const __m128i *>( &tmp1[2] ) ) );
}
template <> vec<int64_t, 4> vec<int64_t, 4>::operator+ (const vec<int64_t, 4> &v) const {
  return vec<int64_t, 4>(  _mm_add_epi64(xmmi[0], v.xmmi[0]) , _mm_add_epi64(xmmi[1], v.xmmi[1]) );
}
template <> vec<int64_t, 4> vec<int64_t, 4>::operator- (const vec<int64_t, 4> &v) const {
  return vec<int64_t, 4>(  _mm_sub_epi64(xmmi[0], v.xmmi[0]), _mm_sub_epi64(xmmi[1], v.xmmi[1]) );
}
/*template <> vec<int64_t, 4> vec<int64_t, 4>::operator/ (const vec<int64_t, 4> &v) const {
  return vec<int64_t, 4>(  _mm_sub_epi64(xmmi[0], v.xmmi[0]) );
}*/
template <> inline void vec<int64_t, 4>::operator*= (const vec<int64_t, 4> &v) {
  /*xmm[0]=  _mm_mul_epi32( xmmi[0], v.xmmi[0] );
  xmm[1]=  _mm_mul_epi32( xmmi[1], v.xmmi[1] );*/
  int64_t tmp1[4];
  int64_t tmp2[4];
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp1[0] ),   xmm[0]);
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp1[2] ),   xmm[1]);
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp2[0] ), v.xmm[0]);
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp2[2] ), v.xmm[1]);
  tmp1[0] = tmp1[0] * tmp2[0];
  tmp1[1] = tmp1[1] * tmp2[1];
  tmp1[2] = tmp1[2] * tmp2[2];
  tmp1[3] = tmp1[3] * tmp2[3];
  xmm[0] = _mm_loadu_si128( reinterpret_cast<const __m128i *>( &tmp1[0] ) ); 
  xmm[1] = _mm_loadu_si128( reinterpret_cast<const __m128i *>( &tmp1[2] ) );
}
template <> inline void vec<int64_t, 4>::operator+= (const vec<int64_t, 4> &v) {
  xmm[0]=  _mm_add_epi64(xmmi[0], v.xmmi[0] );
  xmm[1]=  _mm_add_epi64(xmmi[1], v.xmmi[1] );
}
template <> inline void vec<int64_t, 4>::operator-= (const vec<int64_t, 4> &v) {
  xmm[0]=  _mm_sub_epi64(xmmi[0], v.xmmi[0] );
  xmm[1]=  _mm_sub_epi64(xmmi[1], v.xmmi[1] );
}
/*template <> inline void vec<int64_t, 4>::operator/= (const vec<int64_t, 4> &v) {
  xmm[0]=  _mm_mul_epi32(xmmi[0], v.xmmi[0] );
  xmm[0]=  _mm_mul_epi32(xmmi[0], v.xmmi[0] );
}*/

// Operators,  scalar versions (int64_t , length 4):
template <> vec<int64_t, 4> vec<int64_t, 4>::operator* (const int64_t v) const {
  /*__m128i tmp = _mm_set_epi64x( v, v);
  return vec<int64_t, 4>( _mm_mul_epi32(xmm[0], tmp), _mm_mul_epi32(xmm[1], tmp));*/
  int64_t tmp1[4];
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp1[0] ),   xmm[0]);
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp1[2] ),   xmm[1]);
  tmp1[0] = tmp1[0] * v;
  tmp1[1] = tmp1[1] * v;
  tmp1[2] = tmp1[2] * v;
  tmp1[3] = tmp1[3] * v;
  return vec<int64_t, 4>( _mm_loadu_si128( reinterpret_cast<const __m128i *>( &tmp1[0] ) ), 
                          _mm_loadu_si128( reinterpret_cast<const __m128i *>( &tmp1[2] ) ) );
}
template <> vec<int64_t, 4> vec<int64_t, 4>::operator+ (const int64_t &v) const {
  __m128i tmp = _mm_set_epi64x( v, v);
  return vec<int64_t, 4>( _mm_add_epi64(xmm[0], tmp), _mm_add_epi64(xmm[1], tmp));
}
template <> vec<int64_t, 4> vec<int64_t, 4>::operator- (const int64_t &v) const {
  __m128i tmp = _mm_set_epi64x( v, v);
  return vec<int64_t, 4>( _mm_sub_epi64(xmm[0], tmp), _mm_sub_epi64(xmm[1], tmp));
}
/*template <> vec<int64_t, 4> vec<int64_t, 4>::operator/ (const int64_t &v) const {
  __m128i tmp = _mm_set_epi64x( v, v);
  return vec<int64_t, 4>( _mm_sub_epi64(xmm[0], tmp));
}*/
template <> inline void vec<int64_t, 4>::operator*= (const int64_t &v) {
  /*__m128i tmp = _mm_set_epi64x( v, v);
  xmm[0] = _mm_mul_epi32(xmm[0], tmp);
  xmm[1] = _mm_mul_epi32(xmm[1], tmp);*/
  int64_t tmp1[4];
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp1[0] ),   xmm[0]);
  _mm_storeu_si128(  reinterpret_cast< __m128i *> (&tmp1[2] ),   xmm[1]);
  tmp1[0] = tmp1[0] * v;
  tmp1[1] = tmp1[1] * v;
  tmp1[2] = tmp1[2] * v;
  tmp1[3] = tmp1[3] * v;
  xmm[0] = _mm_loadu_si128( reinterpret_cast<const __m128i *>( &tmp1[0] ) ); 
  xmm[1] = _mm_loadu_si128( reinterpret_cast<const __m128i *>( &tmp1[2] ) );
}
template <> inline void vec<int64_t, 4>::operator+= (const int64_t &v) {
  __m128i tmp = _mm_set_epi64x( v, v);
  xmm[0] = _mm_add_epi64(xmm[0], tmp);
  xmm[1] = _mm_add_epi64(xmm[1], tmp);
}
template <> inline void vec<int64_t, 4>::operator-= (const int64_t &v) {
  __m128i tmp = _mm_set_epi64x( v, v);
  xmm[0] = _mm_sub_epi64(xmm[0], tmp);
  xmm[1] = _mm_sub_epi64(xmm[1], tmp);
}
/*template <> inline void vec<int64_t, 4>::operator/= (const int64_t &v) {
  __m128i tmp = _mm_set_epi64x( v, v);
  xmm[0] = _mm_div_epi32(xmm[0], tmp);
  xmm[0] = _mm_div_epi32(xmm[0], tmp);
}*/
template <> inline vec<int64_t, 4> vec<int64_t, 4>::operator= (const int64_t &v) {
  xmm[0] = _mm_set_epi64x( v, v);
  xmm[1] = xmm[0];
  return vec<int64_t, 4>( xmm[0], xmm[0] );
}



// defined in emm_vec_int64_2.cpp :
//int64_t ALIGN16 negation64[4] = {0x8000000000000000,0x8000000000000000};
template <> vec<int64_t, 4> vec<int64_t, 4>::operator>> (const int shiftcount) const  {
  // shift right not supported for signed int64 values. (ARGH!!, why???)
  __m128i tmp = _mm_load_si128( (__m128i *) &negation64[0] );
  //tmp.m128i_u64[0] = 0x8000000000000000;
  //tmp.m128i_u64[1] = 0x8000000000000000;
  return vec<int64_t, 4>( _mm_sub_epi64( _mm_srli_epi64( _mm_add_epi64( xmmi[0], tmp), shiftcount), _mm_srli_epi64( tmp, shiftcount) ) ,
                            _mm_sub_epi64( _mm_srli_epi64( _mm_add_epi64( xmmi[1], tmp), shiftcount), _mm_srli_epi64( tmp, shiftcount) )  );
}

template <> vec<int64_t, 4> vec<int64_t, 4>::operator<< (const int shiftcount) const  {
  return vec<int64_t, 4>( _mm_slli_epi64(xmmi[0], shiftcount) , _mm_slli_epi64(xmmi[1], shiftcount) );
}
template <> vec<int64_t, 4> vec<int64_t, 4>::operator| (const vec<int64_t, 4> &v) const {
  return vec<int64_t, 4>( _mm_or_si128(xmmi[0], v.xmmi[0]), _mm_or_si128(xmmi[1], v.xmmi[1]) );
}
template <> vec<int64_t, 4> vec<int64_t, 4>::operator& (const vec<int64_t, 4> &v) const {
  return vec<int64_t, 4>( _mm_and_si128(xmmi[0], v.xmmi[0]), _mm_and_si128(xmmi[1], v.xmmi[1]) );
}
#ifdef INCLUDE_SSE4
template <> vec<int64_t, 4> vec<int64_t, 4>::operator>(const vec<int64_t, 4> &v) const  {
  return vec<int64_t, 4>( _mm_cmpgt_epi64(xmmi[0], v.xmmi[0]), _mm_cmpgt_epi64(xmmi[1], v.xmmi[1]) );
}
/* Somehow lt comparison of 64 bit integers is not supported
// obviously !((a>b) | (a==b)) is the same, but much slower.
template <> vec<int64_t, 4> vec<int64_t, 4>::operator<(const vec<int64_t, 4> &v) const  {
  return vec<int64_t, 4>( _mm_cmplt_epi64(xmmi[0], v.xmmi[0]) );
}*/
template <> vec<int64_t, 4> vec<int64_t, 4>::operator==(const vec<int64_t, 4> &v) const {
  return vec<int64_t, 4>( _mm_cmpeq_epi64(xmmi[0], v.xmmi[0]) ,  _mm_cmpeq_epi64(xmmi[1], v.xmmi[1]) );
}
#endif


#ifdef INCLUDE_SSE4
// other functions (min/max ..) (int64, length 4)
inline vec<int64_t, 4> max_bad(const vec<int64_t, 4> &a, const vec<int64_t, 4> &b){
    // _bad : using 32 bit maximum function
  return vec<int64_t,4>(_mm_max_epi32(a.xmmi[0], b.xmmi[0]), _mm_max_epi32(a.xmmi[1], b.xmmi[1]));
}
inline vec<int64_t, 4> min_bad(const vec<int64_t, 4> &a, const vec<int64_t, 4> &b){
    // _bad : using 32 bit maximum function
  return vec<int64_t,4>(_mm_min_epi32(a.xmmi[0], b.xmmi[0]), _mm_min_epi32(a.xmmi[1], b.xmmi[1]));
}
#endif
