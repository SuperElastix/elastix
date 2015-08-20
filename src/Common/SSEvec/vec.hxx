#ifndef EMM_VEC_CPP // only include once.
#define EMM_VEC_CPP
/* This file provides a vector class with compile time known vector length. 
 * This file is a general case and debug-safe version of emm_vec.hxx. 
 * The 'vec' class provided in this file should have the same interface as the emm_vec.hxx version that is prefered for release builds. 
 * See help in that file for full interface description.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Date 20-8-2015
 */


#pragma message("vec.hxx is included. Use this file over emm_vec.hxx only for debugging purposes or when compiling on a platform not supported by emm_vec.hxx.")

#include <iterator>

#define REGISTER_NUM_BYTES sizeof(double)

// ******  main vector definition ************************************************

template <typename T, int vlen> struct vec {
  typedef T value_type;
  enum {length = vlen};
  enum {naturalLen = vlen} ;
  enum {supported = true };

  T xmm[vlen];

  // Externally callable constructors:   
  explicit inline vec (const T *v) {
    for (int i = 0 ; i < vlen ; ++i ) {
      xmm[i] = v[i];
    }
  }
  template <typename int_t> explicit vec (const T *v, int_t stride) {
    for (int i = 0 ; i < vlen ; ++i ) {
      xmm[i] = v[i*stride];
    }
  }
  static inline vec loada( const T *v ) {
    return vec( v ); // we don't know about alignment of the datatype T, so use normal constructor. 
  }

  explicit inline vec (const T v) {
    for (int i = 0 ; i < vlen ; ++i ) {
      xmm[i] = v;
    }
  }
  static inline vec zero();
  explicit vec () {}; //  default constructor, no initialization is performed so it contains garbage !!!.


  // reinterpret cast:
  template < typename ToType > inline vec< ToType, (vlen*(sizeof (T))/(sizeof (ToType))) > reinterpret()
  {
    return vec<ToType, vlen*(sizeof (T))/(sizeof (ToType)) >( xmm , vlen*(sizeof (T))/(sizeof (ToType)));
  }

  // Type conversion routine:
  template <typename From, int vlen> explicit inline vec(const vec<From, vlen> &v) {
    for (int i = 0 ; i < vlen ; ++i ) {
      xmm[i] = (T) v.xmm[i];
    }
  }

  // Store to memory routines:   (Guardpointer checked versions of these are defined in "emm_vec_internal.hxx")
  inline void store( T *v ) {  // store unaligned.
    for (int i = 0 ; i < vlen ; ++i ) {
      v[i] =  xmm[i];
    }
  }
  inline void storea( T * v) { // store aligned.
    store( v );
  }
  template <typename int_t> inline void store( T *v, int_t stride ) {
    for (int i = 0 ; i < vlen ; ++i ) { 
      v[ i * stride ] =  xmm[ i ];
    }
  }

  // Define the operators that we support:
  // vector versions:
  inline vec operator* (const vec<T, vlen> &v) const {
    T t[ vlen ];
    for (int i = 0 ; i < vlen ; ++i ) {
      t[i] =  xmm[i] * v.xmm[i];
    }
    return vec< T, vlen >( t );
  };
  inline vec operator+ (const vec<T, vlen> &v) const{
    T t[ vlen ];
    for (int i = 0 ; i < vlen ; ++i ) {
      t[i] =  xmm[i] + v.xmm[i];
    }
    return vec< T, vlen >( t );
  };
  inline vec operator- (const vec<T, vlen> &v) const{
    T t[ vlen ];
    for (int i = 0 ; i < vlen ; ++i ) {
      t[i] =  xmm[i] - v.xmm[i];
    }
    return vec< T, vlen >( t );
  };
  inline vec operator/ (const vec<T, vlen> &v) const{
    T t[ vlen ];
    for (int i = 0 ; i < vlen ; ++i ) {
      t[i] =  xmm[i] / v.xmm[i];
    }
    return vec< T, vlen >( t );
  };
  inline void operator*= (const vec<T, vlen> &v){
    for (int i = 0 ; i < vlen ; ++i ) {
      xmm[i] *= v.xmm[i];
    }
  };
  inline void operator+= (const vec<T, vlen> &v){
    for (int i = 0 ; i < vlen ; ++i ) {
      xmm[i] += v.xmm[i];
    }
  };
  inline void operator-= (const vec<T, vlen> &v){
    for (int i = 0 ; i < vlen ; ++i ) {
      xmm[i] -= v.xmm[i];
    }
  };
  inline void operator/= (const vec<T, vlen> &v){
    for (int i = 0 ; i < vlen ; ++i ) {
      xmm[i] /= v.xmm[i];
    }
  };

  // vector with scalar versions:
  inline vec operator* (const T v) const{
    T t[ vlen ];
    for (int i = 0 ; i < vlen ; ++i ) {
      t[i] =  xmm[i] * v;
    }
    return vec< T, vlen >( t );
  };
  inline vec operator+ (const T &v) const{
    T t[ vlen ];
    for (int i = 0 ; i < vlen ; ++i ) {
      t[i] =  xmm[i] + v;
    }
    return vec< T, vlen >( t );
  };
  inline vec operator- (const T &v) const{
    T t[ vlen ];
    for (int i = 0 ; i < vlen ; ++i ) {
      t[i] =  xmm[i] - v;
    }
    return vec< T, vlen >( t);
  };
  inline vec operator/ (const T &v) const{
    T t[ vlen ];
    for (int i = 0 ; i < vlen ; ++i ) {
      t[i] =  xmm[i] / v;
    }
    return vec< T, vlen >( t);
  };
  inline void operator*= (const T &v){
    for (int i = 0 ; i < vlen ; ++i ) {
      xmm[i] *= v;
    }
  };
  inline void operator+= (const T &v){
    for (int i = 0 ; i < vlen ; ++i ) {
      xmm[i] += v;
    }
  };
  inline void operator-= (const T &v){
    for (int i = 0 ; i < vlen ; ++i ) {
      xmm[i] -= v;
    }
  };
  inline void operator/= (const T &v){
    for (int i = 0 ; i < vlen ; ++i ) {
      xmm[i] /= v;
    }
  };
  inline vec operator= (const T &v){
    for (int i = 0 ; i < vlen ; ++i ) {
      xmm[i] = v;
    };
    return vec< T, vlen> ( xmm );
  };

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
  inline vec operator& (const vec<T, vlen> &v) const; */


  // other operations:
  inline vec rep(int idx) const { // Replicate vec[idx]; use only for compile time constant idx.
    return vec<T, vlen>( xmm[ idx ] );
  }
  // inline vec insert( const vec &v, const vec &mask );
  inline void set( int idx, const T & value ){ // set specific entry of the vector, use only for compile time constant idx.
    xmm[idx] = value;
  }
  // inline static vec signmask();
  // template <typename From, int vlenFrom> inline void scale_unsafe(const vec<From, vlenFrom> &v);

};

// Unary minus:
template<typename T, int vlen> vec<T, vlen> operator-(const vec<T, vlen> & v) {
  for (int i = 0 ; i < vlen ; ++i ) {
      xmm[i] = -xmm[i];
  };
}
// zero:
template<typename T> T zero() {
  for (int i = 0 ; i < vlen ; ++i ) {
      xmm[i] = 0;
  };
}

// General functions:
/*
template <typename T, int vlen> vec<T, vlen> vec<T, vlen>::insert( const vec<T, vlen> &b, const vec<T, vlen> &mask ) {
  return andnot(mask, *this ) | (b & mask);
};*/

template <typename T>
inline void conditional_swap(T &a, T &b, const T condition){
/* for all bits in condition that are 1 (i.e. when condition is true)
   the corresponding bits of a and b are swapped.
   Example:
      conditional_swap(a,b, a>b)
   performs for all elements in a and b: if a[i]>b[i] then swap(a[i],b[i])
   so after evaluation of this function a[i]<=b[i]
   */

    T tmp = ((a^b)& condition);
    // if true (i.e. all bits set)  : tmp = a xor b
    //                   else : tmp = 0
    a = (a^tmp);
    b = (b^tmp);
}

template <typename T>  inline T sqr(T a) {
/* Sqr: compute square. For vector arguments, the element wise square is computed.
    Created by Dirk Poot, Erasmus MC
   */
   return a*a;
};

template < typename T> inline T sum( T a) {
  return a;
};


// Test template
template < typename T> struct IS_vec {
	enum { TF = false };
};
template < typename T, int vlen > struct IS_vec< vec<T, vlen > > {
	enum { TF  = true };
};

#endif
