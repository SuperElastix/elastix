/* This file provides the implementation of the  vec< double , 4>  type
 * Only for including in emm_vec.hxx.
 * Don't include anywhere else.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 19-4-2012
 */


// load vector:
template<>  vec<double, 4>::vec(const double *v) {
  if (__alignof(v)>=16) {
      xmmd[0] = _mm_load_pd(v );
      xmmd[1] = _mm_load_pd(v + 2);
  } else {
      xmmd[0] = _mm_loadu_pd(v );
      xmmd[1] = _mm_loadu_pd(v + 2);
  }
}
template<> template<typename int_t> vec<double, 4>::vec(const double *v, int_t stride) {
      xmmd[0] = _mm_load_sd(v );
      xmmd[0] = _mm_loadh_pd(xmmd[0], v+stride );
      xmmd[1] = _mm_load_sd(v + 2*stride);
      xmmd[1] = _mm_loadh_pd(xmmd[1], v+3*stride );
}
template<>  vec<double, 4>::vec(const double v) {
      xmmd[0] = _mm_load_sd(&v );
//    xmmd[0].m128d_f64[0] = v;
    xmmd[0] = _mm_unpacklo_pd(xmmd[0], xmmd[0]);
      xmmd[1] = xmmd[0];
}
template<> vec<double, 4>  vec<double, 4>::loada( const double *v ) {
  return vec<double,4>( _mm_load_pd(v ),_mm_load_pd(v + 2 ));
}
//create as zero vector:
// template <> vec<double, 4> vec<double, 4>::zero () {
//  return vec<double,4>(_mm_setzero_pd(), _mm_setzero_pd());
// }
template <> vec<double, 4> zero () {
    return vec<double,4>(_mm_setzero_pd(), _mm_setzero_pd());
}

// Store functions:
template <> void vec<double, 4>::store(double *v) {
  _mm_storeu_pd( (v    ), xmmd[0]);
  _mm_storeu_pd( (v + 2), xmmd[1]);
}
template <> void vec<double, 4>::storea(double *v) {
  _mm_store_pd( (v    ), xmmd[0]);
  _mm_store_pd( (v + 2), xmmd[1]);
}
template<> template<typename int_t> void vec<double, 4>::store(double *v, int_t stride) {
  _mm_store_sd(  (v    ), xmmd[0]);
  _mm_storeh_pd( (v +   stride), xmmd[0]);
  _mm_store_sd(  (v + 2*stride), xmmd[1]);
  _mm_storeh_pd( (v + 3*stride), xmmd[1]);
}

// Type conversion constructors; convert this file's type to different types
template<> template<> inline vec<int32_t, 4>::vec(const vec<double, 4> &v) {
  xmmi[0] =     _mm_cvtpd_epi32(v.xmmd[0]);
  __m128i tmp = _mm_cvtpd_epi32(v.xmmd[1]);
  xmmi[0] = _mm_unpacklo_epi64(xmmi[0], tmp);
}

template<> template<> inline vec<float, 4>::vec(const vec<double, 4> &v) {
  xmm[0] = _mm_cvtpd_ps(v.xmmd[0]);
    __m128 tmp = _mm_cvtpd_ps(v.xmmd[1]);
  xmm[0] = _mm_shuffle_ps(xmm[0], tmp, _MM_SHUFFLE(1, 0, 1, 0));
}
#ifdef INCLUDE_SSE4
template<> template<> inline vec<int64_t, 4>::vec(const vec<double, 4> &v) {
  xmmi[0] = _mm_cvtpd_epi32(v.xmmd[0]);
  xmmi[0] = _mm_cvtepi32_epi64(xmmi[0]);
  xmmi[1] = _mm_cvtpd_epi32(v.xmmd[1]);
  xmmi[1] = _mm_cvtepi32_epi64(xmmi[1]);
}
#endif

// Operators, Specialized versions (double precision, length 4):
template <> vec<double, 4> vec<double, 4>::operator* (const vec<double,4> &v) const {
  return vec<double,4>(_mm_mul_pd(xmmd[0], v.xmmd[0]), _mm_mul_pd(xmmd[1], v.xmmd[1]));
}
template <> vec<double, 4> vec<double, 4>::operator+ (const vec<double,4> &v) const {
  return vec<double,4>(_mm_add_pd(xmmd[0], v.xmmd[0]), _mm_add_pd(xmmd[1], v.xmmd[1]));
}
template <> vec<double, 4> vec<double, 4>::operator- (const vec<double,4> &v) const {
  return vec<double,4>(_mm_sub_pd(xmmd[0], v.xmmd[0]), _mm_sub_pd(xmmd[1], v.xmmd[1]));
}
template <> vec<double, 4> vec<double, 4>::operator/ (const vec<double,4> &v) const {
  return vec<double,4>(_mm_div_pd(xmmd[0], v.xmmd[0]), _mm_div_pd(xmmd[1], v.xmmd[1]));
}
template <> inline void vec<double, 4>::operator*= (const vec<double, 4> &v) {
  xmmd[0] = _mm_mul_pd(xmmd[0], v.xmmd[0]);
  xmmd[1] = _mm_mul_pd(xmmd[1], v.xmmd[1]);
}
template <> inline void vec<double, 4>::operator+= (const vec<double, 4> &v) {
  xmmd[0] = _mm_add_pd(xmmd[0], v.xmmd[0]);
  xmmd[1] = _mm_add_pd(xmmd[1], v.xmmd[1]);
}
template <> inline void vec<double, 4>::operator-= (const vec<double, 4> &v) {
  xmmd[0] = _mm_sub_pd(xmmd[0], v.xmmd[0]);
  xmmd[1] = _mm_sub_pd(xmmd[1], v.xmmd[1]);
}
template <> inline void vec<double, 4>::operator/= (const vec<double, 4> &v) {
  xmmd[0] = _mm_div_pd(xmmd[0], v.xmmd[0]);
  xmmd[1] = _mm_div_pd(xmmd[1], v.xmmd[1]);
}
// unary minus:
template<> vec<double, 4> operator-(const vec<double, 4> & v) {
    return vec<double,4>( _mm_sub_pd( _mm_setzero_pd(),v.xmmd[0]),_mm_sub_pd( _mm_setzero_pd(),v.xmmd[1]));
}

//  Operators, scalar versions (double, length 4):
template <> vec<double, 4> vec<double, 4>::operator* (const double v) const {
  //__m128d tmp = _mm_load_sd(&v);
  //tmp = _mm_unpacklo_pd(tmp, tmp);
  __m128d tmp =  _mm_set1_pd(v); //_mm_loaddup_pd ( &v );
  return vec<double,4>(_mm_mul_pd(xmmd[0], tmp), _mm_mul_pd(xmmd[1], tmp));
}
template <> vec<double, 4> vec<double, 4>::operator+ (const double &v) const  {
  __m128d tmp = _mm_load_sd(&v);
  tmp = _mm_unpacklo_pd(tmp, tmp);
  return vec<double,4>(_mm_add_pd(xmmd[0], tmp), _mm_add_pd(xmmd[1], tmp));
}
template <> vec<double, 4> vec<double, 4>::operator- (const double &v) const  {
  __m128d tmp = _mm_load_sd(&v);
  tmp = _mm_unpacklo_pd(tmp, tmp);
  return vec<double,4>(_mm_sub_pd(xmmd[0], tmp), _mm_sub_pd(xmmd[1], tmp));
}
template <> vec<double, 4> vec<double, 4>::operator/ (const double &v) const  {
  __m128d tmp = _mm_load_sd(&v);
  tmp = _mm_unpacklo_pd(tmp, tmp);
  return vec<double,4>(_mm_div_pd(xmmd[0], tmp), _mm_div_pd(xmmd[1], tmp));
}
template <> inline void vec<double, 4>::operator*= (const double &v) {
  __m128d tmp = _mm_load_sd(&v);
  tmp = _mm_unpacklo_pd(tmp, tmp);
  xmmd[0] = _mm_mul_pd(xmmd[0], tmp);
  xmmd[1] = _mm_mul_pd(xmmd[1], tmp);
}
template <> inline void vec<double, 4>::operator+= (const double &v) {
  __m128d tmp = _mm_load_sd(&v);
  tmp = _mm_unpacklo_pd(tmp, tmp);
  xmmd[0] = _mm_add_pd(xmmd[0], tmp);
  xmmd[1] = _mm_add_pd(xmmd[1], tmp);
}
template <> inline void vec<double, 4>::operator-= (const double &v) {
  __m128d tmp = _mm_load_sd(&v);
  tmp = _mm_unpacklo_pd(tmp, tmp);
  xmmd[0] = _mm_sub_pd(xmmd[0], tmp);
  xmmd[1] = _mm_sub_pd(xmmd[1], tmp);
}
template <> inline void vec<double, 4>::operator/= (const double &v) {
  __m128d tmp = _mm_load_sd(&v);
  tmp = _mm_unpacklo_pd(tmp, tmp);
  xmmd[0] = _mm_div_pd(xmmd[0], tmp);
  xmmd[1] = _mm_div_pd(xmmd[1], tmp);
}

// repeat element
template <> vec<double, 4> vec<double, 4>::rep (int idx) const  {
  __m128d tmp = ((idx & 2) ? xmmd[1] : xmmd[0]);
  tmp = ( (idx &1 ) ? _mm_unpackhi_pd(tmp, tmp) : _mm_unpacklo_pd(tmp, tmp));
  return vec<double,4>(tmp, tmp);
}
//other members
template<> inline void vec<double,4>::set( int idx, const double &value ) {
    if (idx<2) {
        if (idx==0) {
            xmmd[0] = _mm_loadl_pd(xmmd[0],&value);
        } else {
            xmmd[0] = _mm_loadh_pd(xmmd[0],&value);
        }
    } else {
        if (idx==2) {
            xmmd[1] = _mm_loadl_pd(xmmd[1],&value);
        } else {
            xmmd[1] = _mm_loadh_pd(xmmd[1],&value);
        }
    }
}


// other functions (min/max ..) (double, length 4)
inline vec<double, 4> max(const vec<double, 4> &a, const vec<double, 4> &b){
  return vec<double,4>(_mm_max_pd(a.xmmd[0], b.xmmd[0]), _mm_max_pd(a.xmmd[1], b.xmmd[1]));
}
inline vec<double, 4> min(const vec<double, 4> &a, const vec<double, 4> &b){
  return vec<double,4>(_mm_min_pd(a.xmmd[0], b.xmmd[0]), _mm_min_pd(a.xmmd[1], b.xmmd[1]));
}
#ifdef INCLUDE_SSE3
inline double sum(const vec<double, 4> &v){
  __m128d tmp = _mm_add_pd(v.xmmd[0], v.xmmd[1]);
  tmp = _mm_hadd_pd(tmp,tmp);
  double tmpd;
  _mm_store_sd(&tmpd,tmp);
  return tmpd;
}
#endif
#ifdef INCLUDE_SSE4
inline vec<double, 4> round(const vec<double, 4> &v){
  return vec<double,4>(_mm_round_pd(v.xmmd[0], _MM_FROUND_TO_NEAREST_INT ), _mm_round_pd(v.xmmd[1], _MM_FROUND_TO_NEAREST_INT ));
}
inline vec<double, 4> ceil(const vec<double, 4> &v){
  return vec<double,4>(_mm_round_pd(v.xmmd[0], _MM_FROUND_TO_POS_INF ), _mm_round_pd(v.xmmd[1], _MM_FROUND_TO_POS_INF ));
}
inline vec<double, 4> floor(const vec<double, 4> &v){
  return vec<double,4>(_mm_round_pd(v.xmmd[0], _MM_FROUND_TO_NEG_INF ), _mm_round_pd(v.xmmd[1], _MM_FROUND_TO_NEG_INF ));
}

#endif

inline vec<double, 4> operator^(const vec<double, 4> &a, const vec<double, 4> &b){
  return vec<double,4>(_mm_xor_pd(a.xmmd[0], b.xmmd[0]),_mm_xor_pd(a.xmmd[1], b.xmmd[1]));
}
template <> template <> inline vec< double, 4> vec<int64_t, 4>::reinterpret() {
    return vec<double, 4>( _mm_castsi128_pd(xmmi[0]), _mm_castsi128_pd(xmmi[1]) );
};
template <> vec<double, 4> vec<double, 4>::operator& (const vec<double, 4> &v) const  {
  return vec<double,4>(_mm_and_pd(xmmd[0], v.xmmd[0]), _mm_and_pd(xmmd[1], v.xmmd[1]));
}
