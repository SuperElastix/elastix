#ifndef EMM_VEC_CPP // only include once.
#define EMM_VEC_CPP
/* This file provides the 'vec' class, which is an overhead free templated wrapper around the xmm intrinsic functions.
 * It should (typically) not be included directly, but by emm_vecptr.hxx. 
 *
 * Operations combining vec's and scalars are supported. Except when explicitly stated, all operations are element wise.
 *
 * Currently mainly single/double precision operations are defined, and some
 * int32 and int64 operations.
 * Currently (mainly) vectors of length 2 and 4 are supported.
 *
 * All standard operators are defined ( +, -, *, /, +=, -=, *=, /= )
 * Both with another vec (with the same type & length) as well as with scalars of the correct type
 *
 * Type specification:  vec< data_type, vector_length >
 * Create new        :  vec< data_type, vector_length >( pointer_to_datatype [, stride_in_#elements] )
 *                      vec< data_type, vector_length >( variable_of_datatype )
 * Type conversion (double <-> single) is available.
 * Also (some) conversions between integer types is available. All these conversions use very fast
 * conversions without range checking. 
 *
 * A few special member routines are provided as well.
 * - A (very unsafe) scale routine ( v = v * 2^i, with vec<integer_type, len> i)
 *
 * And finally, some important functions that operate on vector objects:
 * - min, max : element-wise (2 arguments, output vec), and single argument min/max of vector (outputs scalar)
 * - round  : round each element to nearest (even) integer
 * - floor  : round towards - infinity
 * - ceil   : round towards + infinity.
 * - sum    : accumulate values in vector and return scalar of datatype
 * - sqr    : return the element wise square of a vector.
 *
 * NOTE: As it's a lot of work to add all functions for all vec types, some combinations might
 *       still not be implemented. Causing compile time errors if used.
 *     Please feel free to add them; adding a single function is not much work.
 *       Also: some functions are currently only implemented when SSE3 or even SSE4 is enabled.
 * NOTE: Use an optimizing compiler! If you do, using vec introduces no overhead over directly
 *       using the SSE intrinsics. (and makes programming much easier)
 *
 * 
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 20-9-2010
 */


// includes we need to access the intrinsic functions:
#ifdef __SSE3__
  #define INCLUDE_SSE3
#endif
#ifdef __SSE4__
  //#pragma message("SSE4 compilation support detected, enabling SSE4 for vec.")
  #define INCLUDE_SSE4
  #ifndef INCLUDE_SSE3
    #define INCLUDE_SSE3
  #endif
#else
  //#pragma message("No SSE4 support detected.")
#endif

#ifdef INCLUDE_SSE4
  #pragma message("compiling vec with up to SSE4.1 support.")
  #include <smmintrin.h>  // SSE41
#elif defined INCLUDE_SSE3
  #pragma message("compiling vec with up to SSE3 support.")
  #include <pmmintrin.h>  // SSE3
#else
  #pragma message("compiling vec with up to SSE2 support.")
  #include "emmintrin.h"  // SSE2
#endif

#include <stdint.h> // defines standard (fixed width) types. 

// Some compiler specific definitions:
#ifdef _MSC_VER
  // Compiling with MS Visual studio compiler:
  #include <intrin.h>

  #ifdef _M_X64
    #define __x86_64__ _M_X64
  #endif
  #define ALIGN16 __declspec(align(16))
  #define INCLUDE_LONG_VEC_SEPARATELY
  #if _MSC_VER == 1500
    #define BUGGY_MM_EXTRACT_EPI32
    #pragma message("The MS visual studio 9 compiler (Compiler version 15.00) is known to sometimes store vec unaligned which causes segmentation faults.")
  #endif
#else // add check for compiling with gcc.
  // Compiling with gcc
  #define ALIGN16 __attribute__ ((aligned (16)))
#endif

#include <stdarg.h>
#include "standard_templates.hxx"
#ifdef MEX
#include "mex.h"
  #define BAD mexErrMsgTxt("Currently unsupported");
  #define BADARG mexErrMsgTxt("Wrong argument supplied.");
  #define ERRORMSG(arg) mexErrMsgTxt(arg)
  #define WARNMSG(arg) mexWarnMsgTxt(arg)
  #define DISPMSG(arg) mexPrintf(arg)
#else
  #define BAD std::cerr << "ERROR: Currently unsupported." << std::endl;
  #define BADARG std::cerr << "ERROR: Wrong argument supplied." << std::endl;
  #define ERRORMSG(arg) std::cerr << arg << std::endl
  #define WARNMSG(arg) std::cerr << arg << std::endl
  #define DISPMSG(arg) std::cerr << arg << std::endl
#endif

#define ALIGN_VEC ALIGN16
#define REGISTER_NUM_BYTES 16
#include <iterator>
//#include "lineReadWriters.hxx"



// declaration of support info structure. (Fully) Specialize for all supported T and vlen.
template < typename T, int vlen > struct vec_support {
  enum {supported = false};
};
// declaration helper type that specifies which type uses which register type: 
template < typename T > struct get_register_type {
  typedef __m128 xmmType;
};
template <> struct get_register_type< double >  {
  typedef __m128d xmmType;
};
template <> struct get_register_type< int32_t >  {
  typedef __m128i xmmType;
};
template <> struct get_register_type< int64_t >  {
  typedef __m128i xmmType;
};
template <> struct get_register_type< uint32_t >  {
  typedef __m128i xmmType;
};
template <> struct get_register_type< uint64_t >  {
  typedef __m128i xmmType;
};
template <typename T> struct getIntBaseType {
  typedef BAD_TYPE Type;
};
template <> struct getIntBaseType< long > {
  typedef IF< sizeof(long)==sizeof(int32_t) , int32_t, IF< sizeof(long)==sizeof(int64_t), int64_t , BAD_TYPE >::RET >::RET Type;
};
template <> struct getIntBaseType< unsigned long > {
  typedef IF< sizeof(unsigned long)==sizeof(uint32_t) , uint32_t,  IF< sizeof(unsigned long)==sizeof(uint64_t), uint64_t , BAD_TYPE >::RET >::RET Type;
};
/*template <> struct get_register_type< long >  {
  typedef __m128i xmmType;
};
template <> struct get_register_type< ulong >  {
  typedef __m128i xmmType;
};*/
#define xmmi xmm
#define xmmd xmm


// ******  main vector definition ************************************************

template <typename T, int vlen> struct ALIGN_VEC vec {
  typedef T value_type;
  enum {length = vlen};
  enum {naturalLen = (REGISTER_NUM_BYTES/sizeof(T))} ;
  enum {supported = vec_support<T, vlen>::supported };
// read the 'private' fields, 'private' constructors, copy constructor and assignment.
// Not interesting to users of 'vec'
#include "emm_vec_internal.hxx"


  // Externally callable constructors:   (Guardpointer checked versions of these are defined in "emm_vec_internal.hxx")
  explicit inline vec (const T *v);
  template <typename int_t> explicit vec (const T *v, int_t stride);
  static inline vec loada( const T *v );

  explicit inline vec (const T v);
  static inline vec zero();
  explicit vec () {}; //  default constructor, no initialization is performed so it contains garbage !!!.


  // reinterpret cast:
  template < typename ToType > inline vec< ToType, (vlen*(sizeof (T))/(sizeof (ToType))) > reinterpret()
  {
    return vec<ToType, vlen*(sizeof (T))/(sizeof (ToType)) >( xmm , (int) (vlen * (sizeof (T)) /REGISTER_NUM_BYTES));
  }

  // Type conversion routine:
  template <typename From, int vlenFrom> explicit inline vec(const vec<From, vlenFrom> &v);
 /* template <typename From, int vlenFrom> inline vec operator=(const vec<From, vlenFrom> &v) {
    return vec( v ); // use cast to do actual type conversion. 
  };*/

  // Store to memory routines:   (Guardpointer checked versions of these are defined in "emm_vec_internal.hxx")
  inline void store( T *v);  // store unaligned.
  typedef ALIGN16 T Ta;
  inline void storea( Ta * v); // store aligned.
  template <typename int_t> inline void store( T *v, int_t stride);

  // Define the operators that we support:
  // vector versions:
  inline vec operator* (const vec<T, vlen> &v) const;
  inline vec operator+ (const vec<T, vlen> &v) const;
  inline vec operator- (const vec<T, vlen> &v) const;
  inline vec operator/ (const vec<T, vlen> &v) const;
  inline void operator*= (const vec<T, vlen> &v);
  inline void operator+= (const vec<T, vlen> &v);
  inline void operator-= (const vec<T, vlen> &v);
  inline void operator/= (const vec<T, vlen> &v);

  // vector with scalar versions:
  inline vec operator* (const T v) const;
  inline vec operator+ (const T &v) const;
  inline vec operator- (const T &v) const;
  inline vec operator/ (const T &v) const;
  inline void operator*= (const T &v);
  inline void operator+= (const T &v);
  inline void operator-= (const T &v);
  inline void operator/= (const T &v);
  inline vec operator= (const T &v);

  // comparison to bitmask:
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
  inline vec operator& (const vec<T, vlen> &v) const;


  // other operations:
  inline vec rep(int idx) const; // Replicate vec[idx]; use only for compile time constant idx.
  inline vec insert( const vec &v, const vec &mask );
  inline void set( int idx, const T & value ); // set specific entry of the vector, use only for compile time constant idx.
  inline static vec signmask();
  template <typename From, int vlenFrom> inline void scale_unsafe(const vec<From, vlenFrom> &v);

};

// Unary minus:
template<typename T, int vlen> vec<T, vlen> operator-(const vec<T, vlen> & v);
// zero:
template<typename T> T zero();
//template<typename T, int vlen> vec<T,vlen> zero();

// List the vec's that we support:
// (unfortunately, they cannot be in the definition files, as casting may instanciate the templates to early for that)
template<> struct vec_support<double, 2> {
  enum {supported = true };
};
template<> struct vec_support<double, 3> {
  enum {supported = true };
};
template<> struct vec_support<double, 4> {
  enum {supported = true };
};
template<> struct vec_support<double, 8> {
  enum {supported = true };
};
template<> struct vec_support<float, 4> {
  enum {supported = true };
};
template<> struct vec_support<float, 8> {
  enum {supported = true };
};
template<> struct vec_support<int64_t, 2> {
  enum {supported = true };
};
template<> struct vec_support<int64_t, 3> {
  enum {supported = true };
};
template<> struct vec_support<int64_t, 4> {
  enum {supported = true };
};
template<> struct vec_support<uint64_t, 2> {
  enum {supported = true };
};
template<> struct vec_support<uint64_t, 3> {
  enum {supported = true };
};
template<> struct vec_support<uint64_t, 4> {
  enum {supported = true };
};
template<> struct vec_support<int32_t, 2> {
  enum {supported = true };
};
template<> struct vec_support<int32_t, 3> {
  enum {supported = true };
};
template<> struct vec_support<int32_t, 4> {
  enum {supported = true };
};
template<> struct vec_support<uint32_t, 2> {
  enum {supported = true };
};
template<> struct vec_support<uint32_t, 3> {
  enum {supported = true };
};
template<> struct vec_support<uint32_t, 4> {
  enum {supported = true };
};
#ifdef INCLUDE_LONG_VEC_SEPARATELY
template<int vlen> struct vec_support<long, vlen> {
  enum {supported = vec< typename getIntBaseType<long>::Type , vlen>::supported };
};
template<int vlen> struct vec_support<unsigned long, vlen> {
  enum {supported = vec< typename getIntBaseType<unsigned long>::Type , vlen>::supported };
};
#endif

// Include the implementations for the different types and vector lengths.
// In each of these files, the implementation follows:
//  - Load functions (incl. zero)
//  - Store functions
//  - Operators
//  - Other functions (min/max ...)
#include "emm_vec_double2.hxx"
#include "emm_vec_double3.hxx"
#include "emm_vec_double4.hxx"
#include "emm_vec_double8.hxx"
#include "emm_vec_float4.hxx"
#include "emm_vec_float8.hxx"
#include "emm_vec_int64_2.hxx"
#include "emm_vec_int64_3.hxx"
#include "emm_vec_int64_4.hxx"
#include "emm_vec_uint64_2.hxx"
#include "emm_vec_uint64_3.hxx"
#include "emm_vec_uint64_4.hxx"
#include "emm_vec_int32_2.hxx"
#include "emm_vec_int32_3.hxx"
#include "emm_vec_int32_4.hxx"
#include "emm_vec_uint32_2.hxx"
#include "emm_vec_uint32_3.hxx"
#include "emm_vec_uint32_4.hxx"

#ifdef INCLUDE_LONG_VEC_SEPARATELY
  #include "emm_vec_long.hxx"
  #define VECLONGTYPE unsigned long
  #define VECOTHERLONGTYPE long
  #include "emm_vec_long.hxx"
#endif

// Include type conversions (needs constructors of all types to be defined prior)
#include "emm_vec_type_conversions2.hxx"
#include "emm_vec_type_conversions3.hxx"
#include "emm_vec_type_conversions4.hxx"




// General functions:

template <typename T, int vlen> vec<T, vlen> vec<T, vlen>::insert( const vec<T, vlen> &b, const vec<T, vlen> &mask ) {
  return andnot(mask, *this ) | (b & mask);
};

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
	enum { TF  = vec_support<T, vlen>::supported };
};


#undef BAD
#undef BADARG 
#undef ERRORMSG
#undef WARNMSG
#undef DISPMSG
#undef xmmi
#undef xmmd

#ifndef EMM_VECPTR_HXX
  #pragma message("Including emm_vecptr.hxx out of legacy considerations. Please include emm_vecptr.hxx instead of emm_vec.hxx directly.")
  #include "emm_vecptr.hxx"
#endif

#endif
