/* This file provides the implementation of the  vec< long , *>  type
 * Only for including in emm_vec.hxx.
 * Don't include anywhere else.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 10-3-2015
 */

// Specify the type as preprocessor directive instead of template argument to have precise control over the types for which this 
// vec template works. 


#ifndef VECLONGTYPE 
  #define VECLONGTYPE long
#endif

// vec implementation for types that are not defined in terms of one of the fixed size data types. 
// such as 'long' in visual studio. 
// In this implementation it is cast to the matching fixed size type (by getIntBaseType) and all operations are
// delgated to the vec implementation of that type. 
// Note that when using an optimizing compiler, nothing (!) of the class below will end up in the compiled program. 
// all operations will just be single (or a few for longer vectors) SSE instructions and if possible it will be kept in the register file.
template < int vlen> struct vec< VECLONGTYPE, vlen > {
private:
  typedef typename getIntBaseType< VECLONGTYPE >::Type baseIntType; // we (should) make sure that baseIntType and long have the same memory format. Hence we can use reinterpret_cast. 
  typedef vec< baseIntType , vlen> baseclass;
  baseclass vi; 
  explicit vec( const baseclass v ): vi(v) {};
public:
  typedef VECLONGTYPE T;
  typedef T value_type;
  enum {length = vlen};
  enum {naturalLen = (REGISTER_NUM_BYTES/sizeof(T))} ;
  enum {supported = vec_support<T, vlen>::supported };

  explicit vec (const T *v) : vi( reinterpret_cast<const baseIntType *>( v ) ) {};
  template <typename int_t> explicit vec(const T * v, int_t stride) : vi( reinterpret_cast<const baseIntType *>( v ) , stride ) {};

  static vec loada( const T *v ) {
    return vec( vi::loada( reinterpret_cast<const baseIntType *>( v ) ) );
  };
  explicit vec (const T v) : vi( reinterpret_cast<const baseIntType >( v ) ) {};
  static vec zero() { 
    return baseclass::zero();
  };

  // Type conversion routine:
  template <typename From> explicit vec(const vec<From, vlen> &v) : vi( v.vi ) {};
  
  inline void store( T * v) {
    vi.store( reinterpret_cast<baseIntType * >( v ) );
  };
  inline void storea( T * v) {
    vi.storea( reinterpret_cast<baseIntType * >( v ) );
  };
  template <typename int_t> inline void store( T *v, int_t stride) {
    vi.store( reinterpret_cast<baseIntType * >( v ), stride);
  };

  inline vec operator* (const vec<T, vlen> &v) const {
    return vec( vi * v.vi ); };
  inline vec operator+ (const vec<T, vlen> &v) const {
    return vec( vi + v.vi ); };
  inline vec operator- (const vec<T, vlen> &v) const {
    return vec( vi - v.vi ); };
  inline vec operator/ (const vec<T, vlen> &v) const {
    return vec( vi /  v.vi ); };
  inline void operator*= (const vec<T, vlen> &v) {
    vi*=v.vi; };
  inline void operator+= (const vec<T, vlen> &v) {
    vi+= v.vi; };
  inline void operator-= (const vec<T, vlen> &v) {
    vi-= v.vi; };
  inline void operator/= (const vec<T, vlen> &v) {
    vi/= v.vi; };

  // scalar versions:
  inline vec operator* (const T v) const {
    return vec( vi * static_cast<const baseIntType>(v) ); };
  inline vec operator+ (const T &v) const {
    return vec( vi + static_cast<const baseIntType>(v) ); };
  inline vec operator- (const T &v) const {
    return vec( vi - static_cast<const baseIntType>(v) ); };
  inline vec operator/ (const T &v) const {
    return vec( vi / static_cast<const baseIntType>(v) ); };
  inline void operator*= (const T &v) {
    vi *= static_cast<const baseIntType>( v ) ; };
  inline void operator+= (const T &v) {
    vi += static_cast<const baseIntType>( v ) ; }
  inline void operator-= (const T &v) {
    vi -= static_cast<const baseIntType>( v ) ; }
  inline void operator/= (const T &v) {
    vi /= static_cast<const baseIntType>( v ) ; }
  inline vec operator= (const T &v) {
    vi =  static_cast<const baseIntType >( v ) ;
    return vec( vi ); };

  /* // comparison to bitmask:
  inline vec operator> (const vec &v) const;
  inline vec operator>= (const vec &v) const;
  inline vec operator== (const vec &v) const;
  inline vec operator<= (const vec &v) const;
  inline vec operator< (const vec &v) const;
  inline vec operator!= (const vec &v) const;

  // bit wise operators:
  inline vec operator>> (const int shiftcount) const;
  inline vec operator<< (const int shiftcount) const;
  inline vec operator| (const vec<T, vlen> &v) const;
  inline vec operator& (const vec<T, vlen> &v) const;*/
  template <typename, int> friend struct vec;
};

#undef VECLONGTYPE 