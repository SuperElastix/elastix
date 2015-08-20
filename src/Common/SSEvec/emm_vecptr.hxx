#ifndef EMM_VECPTR_HXX // only include once.
#define EMM_VECPTR_HXX
/* Helper file that provides a pointer type around vec objects (emm_vec.hxx)
 * Each of the classes below can be treated as a pointer to vec<T, vlen> objects, which are stored in memory
 * as array of elements of T. ptrT is a pointer type to elements of T ( so typically, ptrT =  T * )
 * 
 * For example you could use the following in help text below:
 *   typedef double T;
 *   typedef T *  ptrT;
 *   const int vlen = 2;
 *   typedef vec<T, vlen> value_type ;
 *   ptrT  a ;
 * Notation: { ., .}  is used to denote an vec< T, vlen > object (showing 2 elements, so assuming vlen==2).   
 *   where the . specifies with which value that element of the vec object is filled. 
 *   
 * b = vecptr< ptrT, vlen>( a )            : b[k] == { a[ k*vlen + 0 ] , a[ k*vlen + 1 ] } 
 * b = vecptr_aligned<ptrT, vlen>(a)       : idem as vecptr, but now assuming a[0] is sufficiently aligned for vector instructions (currently 16 byte).  
 *                                           This should result in faster loading/storing (depending on processor architecture). 
 * b = vecTptr<ptrT, vlen>(a)              : b[k] == { a[ k + 0 ] , a[ k + 1 ] } 
 *
 * b = vecptr_step< ptrT, vlen>(a, step)   : b[k] == { a[ k + 0*step ] , a[ k + 1*step ] } 
 * b = vecptr_step2<ptrT, vlen>(a, stepb, stepa) :  b[k] == { a[ k*stepa + 0*stepb ] , a[ k*stepa + 1*stepb ] }
 *
 * b can be used as pointer, so *b is identical to b[0], b can be incremented, added to, etc. 
 * Also you can store values to (use as left hand side in an equation) *b, and b[0], ...
 * 
 * Specifically I use iterator_traits< ptrT >::value_type, with possible const removed, as type T. 
 *
 * For debugging purposes, you might want to include vec.hxx prior to including this file. In that 
 * case a general 'vec' class is loaded which does not use SSE instructions. 
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 20-8-2015
 */

#include "emm_vec.hxx"

// Helper class to provide a reference. Should not be used explicitly. 
template< typename base > class vec_store_helper {
public:
  typedef typename base::value_type value_type;
  typedef typename base::array_type ptrT;
protected:
  ptrT ptr;
public:
  vec_store_helper( ptrT ptr_) : ptr(ptr_) {};

  template< typename otherT > value_type operator*( otherT val ) const {
    return value_type( ptr ) * val;
  }
  template< typename otherT > value_type operator+( otherT val ) const {
    return value_type( ptr ) + val;
  }
  template< typename otherT > value_type operator-( otherT val ) const {
    return value_type( ptr ) - val;
  }
  template< typename otherT > value_type operator/( otherT val ) const {
    return value_type( ptr ) / val;
  }

  inline void operator=(value_type newval) {
    newval.store( ptr );
  }
  inline void operator=(const vec_store_helper<base> newval ) {
	  value_type temp( newval );
	  temp.store( ptr );
  }
  template< typename otherT > inline void operator=(const otherT newval ) {
    value_type temp( newval );
    temp.store( ptr );
  }
  inline void operator+=(value_type newval) {
    ( value_type( ptr ) + newval ).store( ptr );
  }
  inline void operator-=(value_type newval) {
    ( value_type( ptr ) - newval ).store( ptr );
  }
  template< typename otherT > inline void operator*=(otherT newval) {
    ( value_type( ptr ) * newval ).store( ptr );
  }
  inline void operator/=(value_type newval) {
    ( value_type( ptr ) / newval ).store( ptr );
  }
  inline operator value_type() const {
    return value_type( ptr );
  };
  inline base operator&() {
    return base( ptr );
  }
};

template< typename base > class vec_store_helper_aligned  {
public:
  typedef typename base::value_type value_type;
  typedef typename base::array_type ptrT;
protected:
  ptrT ptr;
public:
  vec_store_helper_aligned( ptrT ptr_) : ptr(ptr_) {};

  template< typename otherT > value_type operator*( otherT val ) const {
    return value_type::loada( ptr ) * val;
  }
  template< typename otherT > value_type operator+( otherT val ) const {
    return value_type::loada( ptr ) + val;
  }
  template< typename otherT > value_type operator-( otherT val ) const {
    return value_type::loada( ptr ) - val;
  }
  template< typename otherT > value_type operator/( otherT val ) const {
    return value_type::loada( ptr ) / val;
  }

  inline void operator=(value_type newval) {
    newval.storea( ptr );
  }
/*  inline void operator=(const vec_store_helper_aligned<base> newval ) {
	  value_type temp( newval );
	  temp.storea( ptr );
  }*/
  template< typename otherT > inline void operator=(const otherT newval ) {
    value_type temp( newval );
    temp.storea( ptr );
  }
  inline void operator+=(value_type newval) {
    ( value_type::loada( ptr ) + newval ).storea( ptr );
  }
  inline void operator-=(value_type newval) {
    ( value_type::loada( ptr ) - newval ).storea( ptr );
  }
  template< typename otherT > inline void operator*=(otherT newval) {
    ( value_type::loada( ptr ) * newval ).storea( ptr );
  }
  inline void operator/=(value_type newval) {
    ( value_type::loada( ptr ) / newval ).storea( ptr );
  }
  inline operator value_type() const {
    return value_type::loada( ptr );
  };
  inline base operator&() {
    return base( ptr );
  }
};

template < typename T > class removeConst {
public:
  typedef T type;
};
template < typename T> class removeConst< const T> {
public:
  typedef T type;
};

using std::iterator_traits;

template <typename ptrT, int vlen>  class vecptr {
//  vecptr< T, vlen>
// class that implements a random access iterator to vec< T, vlen> elements
// thus this class represents an array of /pointer to vec<T, vlen> elements
// For example:
//    a = vecptr<T , vlen> ( T  ptr )
//    a[3] returns vec<T , vlen>( T + 3* vlen )
//
// that is from an array of element of T, upon reading this class creates a vec object
// Created by Dirk Poot, Erasmus MC, 5-3-2013
protected:
  ptrT ptr;
  typedef const char * conversionCharType;
public:
  typedef typename iterator_traits< ptrT >::value_type  T;
  typedef vecptr<ptrT, vlen> Self;
  typedef typename std::random_access_iterator_tag iterator_category;
  typedef ptrdiff_t difference_type;
  typedef vec< typename removeConst<T>::type , vlen> value_type;
  typedef ptrT  array_type;
  typedef vec_store_helper<Self> pointer;
  typedef vec_store_helper<Self> reference;
  typedef difference_type distance_type;  // retained
//  typedef lineType< ptrT , typename removeConst<T>::type > elementIteratorType;

  enum {supported = value_type::supported };
  vecptr() {}; // default constructor: unitialized pointer!!!
  vecptr( ptrT ptr_): ptr(ptr_) {};
  inline value_type operator*() const {
    return value_type(ptr);
  };
  inline reference operator*()  {
    return reference(ptr);
  };
  inline value_type operator[](ptrdiff_t index) const {
    return value_type(ptr+index*vlen);
  }
  inline reference operator[](ptrdiff_t index) {
    return reference(ptr+index*vlen);
  };

  inline Self operator+(ptrdiff_t step) const {
    return Self( ptr+ step*vlen);
  };
  inline void operator+=(ptrdiff_t step) {
    ptr+= step*vlen;
  };
  inline Self operator-(ptrdiff_t step) const {
    return Self( ptr-step*vlen);
  };
  inline difference_type operator-( Self other) const {
    return (ptr - other.ptr)/vlen;
  }
  inline Self operator++() {
    return Self( ptr+=vlen);
  };
  inline Self operator--() {
    return Self( ptr-=vlen);
  };
  inline bool operator>=( Self other) {
    return ptr>=other.ptr;
  };
  inline bool operator<( Self other) {
    return ptr<other.ptr;
  };
  ptrT getRawPointer() const {
    return ptr;
  };
  operator conversionCharType() const {
    return (const char *) ptr;
  };
  /*elementIteratorType elementIterator( int k ) {
    return elementIteratorType( ptr+k, vlen );
  }*/
};

template <typename ptrT, int vlen>  class vecptr_aligned :  public vecptr<ptrT, vlen >{
public:
  typedef vecptr< ptrT, vlen > Superclass;
  typedef typename Superclass::T  T;
  typedef vecptr_aligned<ptrT, vlen> Self;
  //typedef typename std::random_access_iterator_tag iterator_category;
  //typedef ptrdiff_t difference_type;
  typedef  typename Superclass::value_type value_type;
  //typedef ptrT array_type;
  typedef vec_store_helper_aligned<Self> pointer;
  typedef vec_store_helper_aligned<Self> reference;
  //typedef difference_type distance_type;  // retained
  //typedef lineType< ptrT , typename removeConst<T>::type > elementIteratorType;

  vecptr_aligned(): Superclass() {}; // default constructor: unitialized pointer!!!
  vecptr_aligned( ptrT ptr_): Superclass(ptr_) {};
  inline value_type operator*() const {
    return value_type::loada(this->ptr);
  };
  inline reference operator*()  {
    return reference(this->ptr);
  };
  inline value_type operator[](ptrdiff_t index) const {
    return value_type::loada( this->ptr + index*vlen );
  }
  inline reference operator[](ptrdiff_t index) {
    return reference( this->ptr + index*vlen );
  };
  inline Self operator+(ptrdiff_t step) const {
    return Self( this->ptr+ step*vlen);
  };
  inline void operator+=(ptrdiff_t step) {
    this->ptr+= step*vlen;
  };
  inline Self operator-(ptrdiff_t step) const {
    return Self( this->ptr - step*vlen);
  };
  /*inline difference_type operator-( Self other) const {
    return (this->ptr - other.ptr)/vlen;
  }*/
  inline Self operator++() {
    return Self( this->ptr += vlen);
  };
  inline Self operator--() {
    return Self( this->ptr -= vlen);
  };
};

template< typename base > class vec_step_store_helper {
public:
  typedef typename base::value_type value_type;
  typedef typename base::array_type ptrT;
private:
  ptrT ptr;
  ptrdiff_t stepb;
public:
  vec_step_store_helper( ptrT ptr_, ptrdiff_t stepb_) : ptr(ptr_) , stepb(stepb_) {};

  template< typename otherT > value_type operator*( otherT val ) const {
    return value_type( ptr , stepb) * val;
  }
  template< typename otherT > value_type operator+( otherT val ) const {
    return value_type( ptr , stepb) + val;
  }
  template< typename otherT > value_type operator-( otherT val ) const {
    return value_type( ptr , stepb) - val;
  }
  template< typename otherT > value_type operator/( otherT val ) const {
    return value_type( ptr , stepb) / val;
  }

  inline void operator=(value_type newval) {
    newval.store( ptr , stepb);
  }
  inline void operator=(const vec_step_store_helper<base> newval ) {
	value_type temp( newval );
	temp.store( ptr , stepb);
  }
  template< typename otherT > inline void operator=(const otherT newval ) {
    value_type temp( newval );
    temp.store( ptr ,stepb );
  }
  template< typename otherT > inline void operator*=(otherT newval) {
    ( value_type( ptr , stepb) * newval ).store( ptr, stepb );
  }
  inline void operator+=(value_type newval) {
    ( value_type( ptr, stepb ) + newval ).store( ptr, stepb );
  }
  inline void operator-=(value_type newval) {
    ( value_type( ptr, stepb ) - newval ).store( ptr , stepb);
  }
  inline void operator/=(value_type newval) {
    ( value_type( ptr, stepb ) / newval ).store( ptr , stepb);
  }
  
  inline operator value_type() const {
    return value_type( ptr , stepb );
  };
  inline base operator&() {
    return base( ptr , stepb );
  }
};
/*
template< typename base > class vec_step_store_helper< base, const typename removeConst< typename base::array_type >::type > {
public:
  typedef typename base::value_type value_type;
  typedef typename base::array_type ptrT;
private:
  ptrT ptr;
  ptrdiff_t stepb;
public:
  vec_step_store_helper( ptrT ptr_, ptrdiff_t stepb_) : ptr(ptr_) , stepb(stepb_) {};
  inline void operator=(value_type newval) {
    newval.store( ptr , stepb);
  }
  inline void operator+=(value_type newval) {
    (value_type( ptr )+newval).store( ptr , stepb);
  }
  inline operator value_type() {
    return value_type( ptr , stepb );
  };
  inline base operator&() {
    return base( ptr , stepb );
  }
};*/

template <typename ptrT, int vlen>  class vecptr_step {
//  vecptr_step< T, vlen>
// class that implements a random access iterator to vec< T, vlen> elements
// that are not stored consequtively.
// thus this class represents an array of /pointer to vec<T, vlen> elements
// For example:
//    a = vecptr_step<T , vlen> ( T * ptr , step)
//    a[3] returns vec<T , vlen>( T + 3 , step )
// WARNING: vecptr_step: increments operate on elements of T (as step is expected to be ~=1)
//
// From an array of element of T, upon reading this class creates a vec object
//
// Created by Dirk Poot, Erasmus MC, 5-3-2013

private:
  ptrT ptr;
  ptrdiff_t stepb;
public:

  typedef typename iterator_traits< ptrT >::value_type  T;
  typedef vecptr_step<ptrT, vlen> Self;
  typedef typename std::random_access_iterator_tag iterator_category;
  typedef ptrdiff_t difference_type;
  typedef vec<typename removeConst<T>::type , vlen> value_type;
  typedef ptrT  array_type;
  typedef vec_step_store_helper<Self> pointer;
  typedef vec_step_store_helper<Self> reference;
  typedef difference_type distance_type;  // retained
  typedef  ptrT  elementIteratorType;


  enum {supported = value_type::supported };

  vecptr_step( ptrT ptr_, ptrdiff_t stepb_) : ptr(ptr_) , stepb(stepb_) {};
  inline value_type operator*() const {
    return value_type(ptr , stepb);
  };
  inline reference operator*()  {
    return reference(ptr, stepb);
  };
  inline value_type operator[](ptrdiff_t index) const {
    return value_type(ptr+index, stepb);
  }
  inline reference operator[](ptrdiff_t index) {
    return reference(ptr+index, stepb);
  };

  inline Self operator+(ptrdiff_t step) const {
    return Self( ptr+ step, stepb);
  };
  inline void operator+=(ptrdiff_t step) {
    ptr += step;
  };
  inline Self operator-(ptrdiff_t step) const {
    return Self( ptr-step , stepb);
  };
  inline Self operator++() {
    return Self( ptr+=1 , stepb);
  };
  inline Self operator--() {
    return Self( ptr-=1 , stepb);
  };
  elementIteratorType elementIterator( int k ) {
    return  ptr+k*stepb ;
  }
};

// WARNING: vecptr_step: increments operate on elements of T (as stepb is expected to be ~=1)
template <typename ptrT, int vlen>  class vecptr_step2 {
private:
  ptrT  ptr;
  ptrdiff_t stepa;
  ptrdiff_t stepb;
public:

  typedef typename iterator_traits< ptrT >::value_type  T;
  typedef vecptr_step2<ptrT, vlen> Self;
  typedef typename std::random_access_iterator_tag iterator_category;
  typedef ptrdiff_t difference_type;
  typedef vec<typename removeConst<T>::type , vlen> value_type;
  typedef ptrT  array_type;
  typedef vec_step_store_helper<Self> reference;
  typedef reference pointer;
  typedef difference_type distance_type;  // retained
//  typedef lineType< ptrT , T> elementIteratorType;
//    typedef CHAR * charptr;
    typedef char * charptr;
  enum {supported = value_type::supported };

  vecptr_step2( ptrT ptr_, ptrdiff_t stepb_, ptrdiff_t stepa_) : ptr(ptr_) , stepb(stepb_) , stepa(stepa_) {};
  // vecptr_step2( base_pointer,  step_for_vec_read+write,  step_for_elements_array )

  inline value_type operator*() const {
    return value_type(ptr , stepb);
  };
  inline reference operator*()  {
    return reference(ptr, stepb);
  };
  inline value_type operator[](ptrdiff_t index) const {
    return value_type(ptr+stepa*index, stepb);
  }
  inline reference operator[](ptrdiff_t index) {
    return reference(ptr+stepa*index, stepb);
  };

  inline Self operator+(ptrdiff_t step) const {
    return Self( ptr+ stepa*step, stepb, stepa);
  };
  inline void operator+=(ptrdiff_t step) {
    ptr += stepa*step;
  };
  inline Self operator-(ptrdiff_t step) const {
    return Self( ptr-stepa*step , stepb, stepa);
  };
  inline Self operator++() {
    return Self( ptr+=stepa , stepb, stepa);
  };
  inline Self operator--() {
    return Self( ptr-=stepa , stepb, stepa);
  };
/*  elementIteratorType elementIterator( int k ) {
    return elementIteratorType( ptr+k*stepb, stepa );
  };*/
  inline operator charptr() {
    return (charptr) ptr;
  }
};

// WARNING: vecTptr : increments operate on elements of T
//        instead of elements of vec<T, vlen>, which would be the most logical behaviour and which is
//        implemented in vecptr .
template <typename ptrT, int vlen>  class vecTptr {
private:
  ptrT ptr;
public:

  typedef typename iterator_traits< ptrT >::value_type  T;
  typedef vecTptr<ptrT, vlen> Self;
  typedef typename std::random_access_iterator_tag iterator_category;
  typedef ptrdiff_t difference_type;
  typedef vec<typename removeConst<T>::type, vlen> value_type;
  typedef ptrT  array_type;
  typedef vec_store_helper<Self> reference;
  typedef reference pointer;
  typedef difference_type distance_type;  // retained
  typedef ptrT elementIteratorType;
    //typedef CHAR * charptr;
    typedef char * charptr;
  enum {supported = value_type::supported };

  vecTptr( ptrT ptr_):ptr(ptr_) {};
  inline value_type operator*() const {
    return vec<T, vlen>(ptr);
  };
  inline reference operator*()  {
    return reference(ptr);
  };
  inline value_type operator[](ptrdiff_t index) const {
    return vec<T, vlen>(ptr+index);
  }
  inline reference operator[](ptrdiff_t index) {
    return reference(ptr+index);
  };

  inline Self operator+(ptrdiff_t step) const {
    return Self( ptr+ step);
  };
  inline void operator+=(ptrdiff_t step) {
    ptr+= step;
  };
  inline Self operator-(ptrdiff_t step) const {
    return Self( ptr-step);
  };
  inline Self operator++() {
    return Self( ptr++);
  };
  inline Self operator--() {
    return Self( ptr--);
  };
  elementIteratorType elementIterator( int k ) {
    return ptr+k;
  }
  inline operator charptr() {
    return (charptr) ptr;
  }
};

template <typename ptrT>  class vec_Type {
    // Helper class to get a good vector type of a pointer type
public:
    typedef ptrT Type;
    enum{vlen=1};
};
template <>  class vec_Type<double *> {
public:
    enum {vlen=4};
    typedef vecptr<double *, vlen>  Type;
};
template <>  class vec_Type<const double *> {
public:
    enum {vlen=4};
    typedef vecptr<const double *, vlen>  Type;
};

template <>  class vec_Type<const float *> {
public:
    enum {vlen=4};
    typedef vecptr<float *, vlen>  Type;
};
template <>  class vec_Type<float *> {
public:
    enum {vlen=4};
    typedef vecptr<float *, vlen>  Type;
};

#ifdef STEP_ITERATORS
template <typename T>  class vec_Type< complex_pointer<T> > {
public:
    enum {vlen= vec_Type<T>::vlen };
    typedef complex_pointer< typename vec_Type<T>::Type >  Type;
};

#endif


template <typename ptrT>  class vec_TypeT {
    // Helper class to get a good vector type of a pointer type
public:
    typedef ptrT Type;
    enum{vlen=1};
};
template <>  class vec_TypeT<double *> {
public:
    enum {vlen=4};
    typedef vecTptr<double *, vlen>  Type;
};
template <>  class vec_TypeT<const double *> {
public:
    enum {vlen=4};
    typedef vecTptr<const double *, vlen>  Type;
};

template <>  class vec_TypeT<const float *> {
public:
    enum {vlen=4};
    typedef vecTptr<float *, vlen>  Type;
};
template <>  class vec_TypeT<float *> {
public:
    enum {vlen=4};
    typedef vecTptr<float *, vlen>  Type;
};

#ifdef STEP_ITERATORS
template <typename T>  class vec_TypeT< complex_pointer<T> > {
public:
    enum {vlen= vec_TypeT<T>::vlen };
    typedef complex_pointer< typename vec_TypeT<T>::Type >  Type;
};

#endif


#endif // end of #ifdef EMM_VECPTR_HXX