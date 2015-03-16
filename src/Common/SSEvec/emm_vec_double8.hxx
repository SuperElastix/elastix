/* This file provides the implementation of the  vec< double , 8>  type
 * Only for including in emm_vec.hxx.
 * Don't include anywhere else.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 7-5-2012
 */


// load vector:
template<>  vec<double, 8>::vec(const double *v) {
  if (__alignof(v)>=16) {
      xmmd[0] = _mm_load_pd(v );
      xmmd[1] = _mm_load_pd(v + 2);
      xmmd[2] = _mm_load_pd(v + 4);
      xmmd[3] = _mm_load_pd(v + 6);
  } else {
      xmmd[0] = _mm_loadu_pd(v );
      xmmd[1] = _mm_loadu_pd(v + 2);
      xmmd[2] = _mm_loadu_pd(v + 4);
      xmmd[3] = _mm_loadu_pd(v + 6);
  }
}
template<> template<typename int_t> vec<double, 8>::vec(const double *v, int_t stride) {
      xmmd[0] = _mm_load_sd( v );          v+=stride;
      xmmd[0] = _mm_loadh_pd(xmmd[0], v ); v+=stride;
      xmmd[1] = _mm_load_sd( v );          v+=stride;
      xmmd[1] = _mm_loadh_pd(xmmd[1], v ); v+=stride;
      xmmd[2] = _mm_load_sd( v );          v+=stride;
      xmmd[2] = _mm_loadh_pd(xmmd[2], v ); v+=stride;
      xmmd[3] = _mm_load_sd( v );          v+=stride;
      xmmd[3] = _mm_loadh_pd(xmmd[3], v );
}
template<>  vec<double, 8>::vec(const double v) {
      xmmd[0] = _mm_load_sd( &v );
    xmmd[0] = _mm_unpacklo_pd(xmmd[0], xmmd[0]);
      xmmd[1] = xmmd[0];
      xmmd[2] = xmmd[0];
      xmmd[3] = xmmd[0];
}

//create as zero vector:
template <> vec<double, 8> vec<double, 8>::zero () {
  return vec<double,8>(_mm_setzero_pd(), _mm_setzero_pd(), _mm_setzero_pd(), _mm_setzero_pd());
}

// Store functions:
template <> void vec<double, 8>::store(double *v) {
  _mm_storeu_pd( v , xmmd[0]); v+=2;
  _mm_storeu_pd( v , xmmd[1]); v+=2;
  _mm_storeu_pd( v , xmmd[2]); v+=2;
  _mm_storeu_pd( v , xmmd[3]);
}
template <> void vec<double, 8>::storea(double *v) {
  _mm_store_pd( v , xmmd[0]); v+=2;
  _mm_store_pd( v , xmmd[1]); v+=2;
  _mm_store_pd( v , xmmd[2]); v+=2;
  _mm_store_pd( v , xmmd[3]);
}
template<> template<typename int_t> void vec<double, 8>::store(double *v, int_t stride) {
  _mm_store_sd(  v , xmmd[0]); v+=stride;
  _mm_storeh_pd( v , xmmd[0]); v+=stride;
  _mm_store_sd(  v , xmmd[1]); v+=stride;
  _mm_storeh_pd( v , xmmd[1]); v+=stride;
  _mm_store_sd(  v , xmmd[2]); v+=stride;
  _mm_storeh_pd( v , xmmd[2]); v+=stride;
  _mm_store_sd(  v , xmmd[3]); v+=stride;
  _mm_storeh_pd( v , xmmd[3]);
}

// Type conversion constructors; convert this file's type to different types
// TODO: copy from vec<double,2>


// Operators, Specialized versions (double precision, length 8):
template <> vec<double, 8> vec<double, 8>::operator* (const vec<double,8> &v) const {
  return vec<double,8>( _mm_mul_pd(xmmd[0], v.xmmd[0]),
              _mm_mul_pd(xmmd[1], v.xmmd[1]),
              _mm_mul_pd(xmmd[2], v.xmmd[2]),
              _mm_mul_pd(xmmd[3], v.xmmd[3]));
}
template <> vec<double, 8> vec<double, 8>::operator+ (const vec<double,8> &v) const {
  return vec<double,8>( _mm_add_pd(xmmd[0], v.xmmd[0]),
              _mm_add_pd(xmmd[1], v.xmmd[1]),
              _mm_add_pd(xmmd[2], v.xmmd[2]),
              _mm_add_pd(xmmd[3], v.xmmd[3]));
}
template <> vec<double, 8> vec<double, 8>::operator- (const vec<double,8> &v) const {
  return vec<double,8>( _mm_sub_pd(xmmd[0], v.xmmd[0]),
              _mm_sub_pd(xmmd[1], v.xmmd[1]),
              _mm_sub_pd(xmmd[2], v.xmmd[2]),
              _mm_sub_pd(xmmd[3], v.xmmd[3]));
}
template <> vec<double, 8> vec<double, 8>::operator/ (const vec<double,8> &v) const {
  return vec<double,8>( _mm_div_pd(xmmd[0], v.xmmd[0]),
              _mm_div_pd(xmmd[1], v.xmmd[1]),
              _mm_div_pd(xmmd[2], v.xmmd[2]),
              _mm_div_pd(xmmd[3], v.xmmd[3]));
}
template <> inline void vec<double, 8>::operator*= (const vec<double, 8> &v) {
  xmmd[0] = _mm_mul_pd(xmmd[0], v.xmmd[0]);
  xmmd[1] = _mm_mul_pd(xmmd[1], v.xmmd[1]);
  xmmd[2] = _mm_mul_pd(xmmd[2], v.xmmd[2]);
  xmmd[3] = _mm_mul_pd(xmmd[3], v.xmmd[3]);
}
template <> inline void vec<double, 8>::operator+= (const vec<double, 8> &v) {
  xmmd[0] = _mm_add_pd(xmmd[0], v.xmmd[0]);
  xmmd[1] = _mm_add_pd(xmmd[1], v.xmmd[1]);
  xmmd[2] = _mm_add_pd(xmmd[2], v.xmmd[2]);
  xmmd[3] = _mm_add_pd(xmmd[3], v.xmmd[3]);
}
template <> inline void vec<double, 8>::operator-= (const vec<double, 8> &v) {
  xmmd[0] = _mm_sub_pd(xmmd[0], v.xmmd[0]);
  xmmd[1] = _mm_sub_pd(xmmd[1], v.xmmd[1]);
  xmmd[2] = _mm_sub_pd(xmmd[2], v.xmmd[2]);
  xmmd[3] = _mm_sub_pd(xmmd[3], v.xmmd[3]);
}
template <> inline void vec<double, 8>::operator/= (const vec<double, 8> &v) {
  xmmd[0] = _mm_div_pd(xmmd[0], v.xmmd[0]);
  xmmd[1] = _mm_div_pd(xmmd[1], v.xmmd[1]);
  xmmd[2] = _mm_div_pd(xmmd[2], v.xmmd[2]);
  xmmd[3] = _mm_div_pd(xmmd[3], v.xmmd[3]);
}

//  Operators, scalar versions (double, length 8):
template <> vec<double, 8> vec<double, 8>::operator* (const double v) const {
  //__m128d tmp = _mm_load_sd(&v);
  //tmp = _mm_unpacklo_pd(tmp, tmp);
  __m128d tmp =  _mm_set1_pd(v); //_mm_loaddup_pd ( &v );
  return vec<double,8>(  _mm_mul_pd(xmmd[0], tmp),
               _mm_mul_pd(xmmd[1], tmp),
               _mm_mul_pd(xmmd[2], tmp),
               _mm_mul_pd(xmmd[3], tmp));
}
template <> vec<double, 8> vec<double, 8>::operator+ (const double &v) const  {
  __m128d tmp = _mm_load_sd(&v);
  tmp = _mm_unpacklo_pd(tmp, tmp);
  return vec<double,8>( _mm_add_pd(xmmd[0], tmp),
              _mm_add_pd(xmmd[1], tmp),
              _mm_add_pd(xmmd[2], tmp),
              _mm_add_pd(xmmd[3], tmp));
}
template <> vec<double, 8> vec<double, 8>::operator- (const double &v) const  {
  __m128d tmp = _mm_load_sd(&v);
  tmp = _mm_unpacklo_pd(tmp, tmp);
  return vec<double,8>( _mm_sub_pd(xmmd[0], tmp),
              _mm_sub_pd(xmmd[1], tmp),
              _mm_sub_pd(xmmd[2], tmp),
              _mm_sub_pd(xmmd[3], tmp));
}
template <> vec<double, 8> vec<double, 8>::operator/ (const double &v) const  {
  __m128d tmp = _mm_load_sd(&v);
  tmp = _mm_unpacklo_pd(tmp, tmp);
  return vec<double,8>( _mm_div_pd(xmmd[0], tmp),
              _mm_div_pd(xmmd[1], tmp),
              _mm_div_pd(xmmd[2], tmp),
              _mm_div_pd(xmmd[3], tmp));
}
template <> inline void vec<double, 8>::operator*= (const double &v) {
  __m128d tmp = _mm_load_sd(&v);
  tmp = _mm_unpacklo_pd(tmp, tmp);
  xmmd[0] = _mm_mul_pd(xmmd[0], tmp);
  xmmd[1] = _mm_mul_pd(xmmd[1], tmp);
  xmmd[2] = _mm_mul_pd(xmmd[2], tmp);
  xmmd[3] = _mm_mul_pd(xmmd[3], tmp);
}
template <> inline void vec<double, 8>::operator+= (const double &v) {
  __m128d tmp = _mm_load_sd(&v);
  tmp = _mm_unpacklo_pd(tmp, tmp);
  xmmd[0] = _mm_add_pd(xmmd[0], tmp);
  xmmd[1] = _mm_add_pd(xmmd[1], tmp);
  xmmd[2] = _mm_add_pd(xmmd[2], tmp);
  xmmd[3] = _mm_add_pd(xmmd[3], tmp);
}
template <> inline void vec<double, 8>::operator-= (const double &v) {
  __m128d tmp = _mm_load_sd(&v);
  tmp = _mm_unpacklo_pd(tmp, tmp);
  xmmd[0] = _mm_sub_pd(xmmd[0], tmp);
  xmmd[1] = _mm_sub_pd(xmmd[1], tmp);
  xmmd[2] = _mm_sub_pd(xmmd[2], tmp);
  xmmd[3] = _mm_sub_pd(xmmd[3], tmp);
}
template <> inline void vec<double, 8>::operator/= (const double &v) {
  __m128d tmp = _mm_load_sd(&v);
  tmp = _mm_unpacklo_pd(tmp, tmp);
  xmmd[0] = _mm_div_pd(xmmd[0], tmp);
  xmmd[1] = _mm_div_pd(xmmd[1], tmp);
  xmmd[2] = _mm_div_pd(xmmd[2], tmp);
  xmmd[3] = _mm_div_pd(xmmd[3], tmp);
}
template <> inline vec<double, 8> vec<double, 8>::operator= (const double &v) {
  __m128d tmp = _mm_load_sd(&v);
  xmmd[0] = _mm_unpacklo_pd(tmp, tmp);
  xmmd[3] = xmmd[2] = xmmd[1] = xmmd[0];
  return vec<double, 8>(xmmd[0],xmmd[0],xmmd[0],xmmd[0]);
}

// repeat element
template <> vec<double, 8> vec<double, 8>::rep (int idx) const  {
  __m128d tmp = ((idx & 4) ? ((idx & 2) ? xmmd[3] : xmmd[2])  : ((idx & 2) ? xmmd[1] : xmmd[0]) );
  tmp = ( (idx & 1 ) ? _mm_unpackhi_pd(tmp, tmp) : _mm_unpacklo_pd(tmp, tmp));
  return vec<double,8>(tmp, tmp, tmp, tmp);
}

// other functions (min/max ..) (double, length 8)
inline vec<double, 8> max(const vec<double, 8> &a, const vec<double, 8> &b){
  return vec<double,8>( _mm_max_pd(a.xmmd[0], b.xmmd[0]),
              _mm_max_pd(a.xmmd[1], b.xmmd[1]),
              _mm_max_pd(a.xmmd[2], b.xmmd[2]),
              _mm_max_pd(a.xmmd[3], b.xmmd[3]));
}
inline vec<double, 8> min(const vec<double, 8> &a, const vec<double, 8> &b){
  return vec<double,8>( _mm_min_pd(a.xmmd[0], b.xmmd[0]),
              _mm_min_pd(a.xmmd[1], b.xmmd[1]),
              _mm_min_pd(a.xmmd[2], b.xmmd[2]),
              _mm_min_pd(a.xmmd[3], b.xmmd[3]));
}
#ifdef INCLUDE_SSE3
inline double sum(const vec<double, 8> &v){
  __m128d tmp = _mm_add_pd(v.xmmd[0], v.xmmd[1]);
  __m128d tmp2 = _mm_add_pd(v.xmmd[2], v.xmmd[3]);
  tmp = _mm_add_pd( tmp , tmp2 );
  tmp = _mm_hadd_pd(tmp,tmp);
  double tmpd;
  _mm_store_sd(&tmpd,tmp);
  return tmpd;
}
#endif
#ifdef INCLUDE_SSE4
inline vec<double, 8> round(const vec<double, 8> &v){
  return vec<double,8>( _mm_round_pd(v.xmmd[0], _MM_FROUND_TO_NEAREST_INT ),
                      _mm_round_pd(v.xmmd[1], _MM_FROUND_TO_NEAREST_INT ),
                      _mm_round_pd(v.xmmd[2], _MM_FROUND_TO_NEAREST_INT ),
                      _mm_round_pd(v.xmmd[3], _MM_FROUND_TO_NEAREST_INT ));
}
inline vec<double, 8> ceil(const vec<double, 8> &v){
  return vec<double,8>( _mm_round_pd(v.xmmd[0], _MM_FROUND_TO_POS_INF ),
              _mm_round_pd(v.xmmd[1], _MM_FROUND_TO_POS_INF ),
              _mm_round_pd(v.xmmd[2], _MM_FROUND_TO_POS_INF ),
              _mm_round_pd(v.xmmd[3], _MM_FROUND_TO_POS_INF ));
}
inline vec<double, 8> floor(const vec<double, 8> &v){
  return vec<double,8>( _mm_round_pd(v.xmmd[0], _MM_FROUND_TO_NEG_INF ),
              _mm_round_pd(v.xmmd[1], _MM_FROUND_TO_NEG_INF ),
              _mm_round_pd(v.xmmd[2], _MM_FROUND_TO_NEG_INF ),
              _mm_round_pd(v.xmmd[3], _MM_FROUND_TO_NEG_INF ));
}


#endif

