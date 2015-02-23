/* This file provides the implementation of the  vec< double , 2>  type
 * Only for including in emm_vec.hxx.
 * Don't include anywhere else.
 * 
 * Created by Dirk Poot, Erasmus MC, 
 * Last modified 19-4-2012
 */


// load vector
template<>  vec<double, 2>::vec(const double *v) { 
	if (__alignof(v) >= 16) {
		xmmd[0] = _mm_load_pd(v ); 
	} else { 
		xmmd[0] = _mm_loadu_pd(v ); 
	}
}
template <> template<typename int_t>  vec<double, 2>::vec(const double *v, int_t stride) { 
      xmmd[0] = _mm_load_sd(v ); 
      xmmd[0] = _mm_loadh_pd(xmmd[0], v+stride ); 
}
template<>  vec<double, 2>::vec(const double v) { 
      xmmd[0] = _mm_load_sd(&v ); 
	  xmmd[0] = _mm_unpacklo_pd(xmmd[0], xmmd[0]);
}
template<> vec<double, 2>  vec<double, 2>::loada( const double *v ) {
	return vec<double,2>( _mm_load_pd(v ) ); 
}
//create as zero vector:
template <> vec<double, 2> vec<double, 2>::zero () { 
	return vec<double,2>(_mm_setzero_pd()); 
}

// Store functions:
template <> void vec<double, 2>::store(double *v) { 
	_mm_storeu_pd( (v    ), xmmd[0]);
}
template <> void vec<double, 2>::storea(double *v) {
	_mm_store_pd( (v    ), xmmd[0]);
}
template<> template<typename int_t>  void vec<double, 2>::store(double *v, int_t stride) { 
	_mm_store_sd(  (v    ), xmmd[0]);
	_mm_storeh_pd( (v +   stride), xmmd[0]);
}


// Type conversion constructors; convert this file's type to different types :
#ifdef INCLUDE_SSE4
template<> template<> inline vec<int64_t, 2>::vec(const vec<double, 2> &v) { 
	xmmi[0] = _mm_cvtpd_epi32(v.xmmd[0]); 
	xmmi[0] = _mm_cvtepi32_epi64(xmmi[0]);
}
#endif


// Other members:
template <> template<> void vec<double, 2>::scale_unsafe(const vec<int64_t, 2> &v) { 
    // Computes self = self * 2^v
	// No INF or NAN checks, sign (and exponent) of value corrupted when exponent overflows.
	__m128i tmp = _mm_slli_epi64(v.xmmi[0],52);
	xmmd[0] = _mm_castsi128_pd( _mm_add_epi64(tmp, _mm_castpd_si128( xmmd[0] )) );
}


// Operators, Specialized versions (double precision, length 2):
template <> vec<double, 2> vec<double, 2>::operator* (const vec<double,2> &v) const	{ 
	return vec<double,2>(_mm_mul_pd(xmmd[0], v.xmmd[0])); 
}
template <> vec<double, 2> vec<double, 2>::operator+ (const vec<double,2> &v) const	{ 
	return vec<double,2>(_mm_add_pd(xmmd[0], v.xmmd[0])); 
}
template <> vec<double, 2> vec<double, 2>::operator- (const vec<double,2> &v) const	{ 
	return vec<double,2>(_mm_sub_pd(xmmd[0], v.xmmd[0])); 
}
template <> vec<double, 2> vec<double, 2>::operator/ (const vec<double,2> &v) const	{ 
	return vec<double,2>(_mm_div_pd(xmmd[0], v.xmmd[0])); 
}
template <> inline void vec<double, 2>::operator*= (const vec<double, 2> &v) { 
	xmmd[0] = _mm_mul_pd(xmmd[0], v.xmmd[0]); 
}
template <> inline void vec<double, 2>::operator+= (const vec<double, 2> &v) { 
	xmmd[0] = _mm_add_pd(xmmd[0], v.xmmd[0]); 
}
template <> inline void vec<double, 2>::operator-= (const vec<double, 2> &v) { 
	xmmd[0] = _mm_sub_pd(xmmd[0], v.xmmd[0]); 
}
template <> inline void vec<double, 2>::operator/= (const vec<double, 2> &v) {
	xmmd[0] = _mm_div_pd(xmmd[0], v.xmmd[0]); 
}
template <> vec<double, 2> vec<double, 2>::operator> (const vec<double,2> &v) const	{ 
	return vec<double,2>(_mm_cmpgt_pd(xmmd[0], v.xmmd[0])); 
}
template <> vec<double, 2> vec<double, 2>::operator>= (const vec<double,2> &v) const	{ 
	return vec<double,2>(_mm_cmpge_pd(xmmd[0], v.xmmd[0])); 
}
template <> vec<double, 2> vec<double, 2>::operator== (const vec<double,2> &v) const	{ 
	return vec<double,2>(_mm_cmpeq_pd(xmmd[0], v.xmmd[0])); 
}
template <> vec<double, 2> vec<double, 2>::operator<= (const vec<double,2> &v) const	{ 
	return vec<double,2>(_mm_cmple_pd(xmmd[0], v.xmmd[0])); 
}
template <> vec<double, 2> vec<double, 2>::operator< (const vec<double,2> &v) const	{ 
	return vec<double,2>(_mm_cmplt_pd(xmmd[0], v.xmmd[0])); 
}
template <> vec<double, 2> vec<double, 2>::operator!= (const vec<double,2> &v) const	{ 
	return vec<double,2>(_mm_cmpneq_pd(xmmd[0], v.xmmd[0])); 
}


//  Operators, scalar versions (double, length 2):
template <> vec<double, 2> vec<double, 2>::operator* (const double v) const	{ 
	__m128d tmp = _mm_load_sd(&v);
	tmp = _mm_unpacklo_pd(tmp, tmp);
	return vec<double,2>(_mm_mul_pd(xmmd[0], tmp)); 
}
template <> vec<double, 2> vec<double, 2>::operator+ (const double &v) const	{ 
	__m128d tmp = _mm_load_sd(&v);
	tmp = _mm_unpacklo_pd(tmp, tmp);
	return vec<double,2>(_mm_add_pd(xmmd[0], tmp)); 
}
template <> vec<double, 2> vec<double, 2>::operator- (const double &v) const	{ 
	__m128d tmp = _mm_load_sd(&v);
	tmp = _mm_unpacklo_pd(tmp, tmp);
	return vec<double,2>(_mm_sub_pd(xmmd[0], tmp)); 
}
template <> vec<double, 2> vec<double, 2>::operator/ (const double &v) const	{ 
	__m128d tmp = _mm_load_sd(&v);
	tmp = _mm_unpacklo_pd(tmp, tmp);
	return vec<double,2>(_mm_div_pd(xmmd[0], tmp)); 
}
template <> inline void vec<double, 2>::operator*= (const double &v) { 
	__m128d tmp = _mm_load_sd(&v);
	tmp = _mm_unpacklo_pd(tmp, tmp);
	xmmd[0] = _mm_mul_pd(xmmd[0], tmp); 
}
template <> inline void vec<double, 2>::operator+= (const double &v) { 
	__m128d tmp = _mm_load_sd(&v);
	tmp = _mm_unpacklo_pd(tmp, tmp);
	xmmd[0] = _mm_add_pd(xmmd[0], tmp); 
}
template <> inline void vec<double, 2>::operator-= (const double &v) { 
	__m128d tmp = _mm_load_sd(&v);
	tmp = _mm_unpacklo_pd(tmp, tmp);
	xmmd[0] = _mm_sub_pd(xmmd[0], tmp); 
}
template <> inline void vec<double, 2>::operator/= (const double &v) {
	__m128d tmp = _mm_load_sd(&v);
	tmp = _mm_unpacklo_pd(tmp, tmp);
	xmmd[0] = _mm_div_pd(xmmd[0], tmp); 
}

//other members
template<> inline void vec<double,2>::set( int idx, const double &value ) {
    if (idx==0) {
        xmmd[0] = _mm_loadl_pd(xmmd[0],&value);
    } else {
        xmmd[0] = _mm_loadh_pd(xmmd[0],&value);
    }
}

// other functions (min/max ..) (double, length 2)
inline vec<double, 2> max(const vec<double, 2> &a, const vec<double, 2> &b){
	return vec<double,2>(_mm_max_pd(a.xmmd[0], b.xmmd[0])); 
}
inline vec<double, 2> min(const vec<double, 2> &a, const vec<double, 2> &b){
	return vec<double,2>(_mm_min_pd(a.xmmd[0], b.xmmd[0])); 
}
#ifdef INCLUDE_SSE3
inline double sum(const vec<double, 2> &v){
	__m128d tmp = _mm_hadd_pd(v.xmmd[0],v.xmmd[0]);
	double tmpd;
	_mm_store_sd(&tmpd,tmp);
	return tmpd; 
}
#endif
#ifdef INCLUDE_SSE4
inline vec<double, 2> round(const vec<double, 2> &v){
	return vec<double,2>(_mm_round_pd(v.xmmd[0], _MM_FROUND_TO_NEAREST_INT )); 
}
inline vec<double, 2> ceil(const vec<double, 2> &v){
	return vec<double,2>(_mm_round_pd(v.xmmd[0], _MM_FROUND_TO_POS_INF )); 
}
inline vec<double, 2> floor(const vec<double, 2> &v){
	return vec<double,2>(_mm_round_pd(v.xmmd[0], _MM_FROUND_TO_NEG_INF )); 
}
#endif


#ifdef __x86_64__ 
inline vec<double, 2> abs(const vec<double, 2> &a){
/*#ifdef INCLUDE_SSE4
	__m128i tmp;
	tmp =  _mm_srl_epi64( _mm_cmpeq_epi64(tmp,tmp) , 1);
#else */
    __m128i tmp = _mm_set1_epi64x( (0x7FFFFFFFFFFFFFFF) );
//#endif
	return vec<double,2>(_mm_and_pd(a.xmmd[0], _mm_castsi128_pd(tmp))); 
}
#endif
inline vec<double, 2> operator^(const vec<double, 2> &a, const vec<double, 2> &b){
	return vec<double,2>(_mm_xor_pd(a.xmmd[0], b.xmmd[0])); 
}
/*inline vec<double, 2> operator^(const vec<double, 2> &a, const vec<int64_t, 2> &b){
	return vec<double,2>(_mm_xor_pd(a.xmmd[0], _mm_castsi128_pd(b.xmmi[0]))); 
}*/

template <> template <> inline vec< double, 2> vec<int64_t, 2>::reinterpret() {
		return vec<double, 2>( _mm_castsi128_pd(xmmi[0]) );
};

template <> vec<double, 2> vec<double, 2>::operator& (const vec<double, 2> &v) const	{ 
	return vec<double,2>(_mm_and_pd(xmmd[0], v.xmmd[0])); 
}
template <> vec<double, 2> vec<double, 2>::operator| (const vec<double, 2> &v) const	{ 
	return vec<double,2>(_mm_or_pd(xmmd[0], v.xmmd[0])); 
}
inline vec<double, 2> andnot(const vec<double, 2> &a, const vec<double, 2> &b){
	return vec<double,2>(_mm_andnot_pd(a.xmmd[0], b.xmmd[0])); 
}




