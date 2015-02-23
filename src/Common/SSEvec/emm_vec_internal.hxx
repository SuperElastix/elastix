/* This file provides some internal fields and constructors
 * for the vec type. As these are not meant to be used outside
 * they are placed in this separate file to ease the reading
 * of the main declaration.
 * Only for including in the vec<> definition in emm_vec.hxx.
 * Don't include anywhere else.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 19-4-2012
 */
  typedef typename get_register_type<T>::xmmType xmmType;
    //Internal data fields:
  xmmType xmm[(sizeof(T)*vlen + REGISTER_NUM_BYTES-1)/REGISTER_NUM_BYTES]; // support partially filled registers. division truncates!
/*  __m128i xmmi[sizeof(T)*vlen/REGISTER_NUM_BYTES];
  __m128 xmm[sizeof(T)*vlen/REGISTER_NUM_BYTES];
  __m128d xmmd[sizeof(T)*vlen/REGISTER_NUM_BYTES]; */

   // Internal constructors:
  explicit vec(xmmType in1) {
    if ((sizeof(T)*vlen + REGISTER_NUM_BYTES-1)/REGISTER_NUM_BYTES != 1)
      BAD;
    xmm[0] = in1;
  }
  explicit vec(xmmType in1, xmmType in2) {
    if ((sizeof(T)*vlen + REGISTER_NUM_BYTES-1)/REGISTER_NUM_BYTES !=2 )
      BAD;
    xmm[0] = in1;
    xmm[1] = in2;
  }
  explicit vec(xmmType in1, xmmType in2, xmmType in3, xmmType in4) {
    if ((sizeof(T)*vlen + REGISTER_NUM_BYTES-1)/REGISTER_NUM_BYTES !=4 )
      BAD;
    xmm[0] = in1;
    xmm[1] = in2;
    xmm[2] = in3;
    xmm[3] = in4;
  }
  /*explicit vec(__m128 in1) {
    if (vlen * sizeof(T) !=1 *REGISTER_NUM_BYTES)
      BAD;
    xmm[0] = in1;
  }
  explicit vec(__m128 in1, __m128 in2) {
    if (vlen * sizeof(T) !=2 *REGISTER_NUM_BYTES)
      BAD;
    xmm[0] = in1;
    xmm[1] = in2;
  }
  explicit vec(__m128i in1) {
    if (vlen * sizeof(T) !=1 *REGISTER_NUM_BYTES)
      BAD;
    xmmi[0] = in1;
  }
  explicit vec(__m128i in1, __m128i in2) {
    if (vlen * sizeof(T) !=2 *REGISTER_NUM_BYTES)
      BAD;
    xmmi[0] = in1;
    xmmi[1] = in2;
  }
  explicit vec(__m128d in1) {
    if (vlen * sizeof(T) !=1 *REGISTER_NUM_BYTES)
      BAD;
    xmmd[0] =  in1;
  }
  explicit vec(__m128d in1, __m128d in2) {
    if (vlen * sizeof(T) !=2 *REGISTER_NUM_BYTES)
      BAD;
    xmmd[0] =  in1;
    xmmd[1] =  in2;
  }*/
    explicit vec(__m128 in[], const int arraylen) {
    if ((sizeof(T)*vlen + REGISTER_NUM_BYTES-1)/REGISTER_NUM_BYTES != arraylen)
      BADARG;
    xmm[0] = in[0];
    if (vlen * sizeof(T) > 1 * REGISTER_NUM_BYTES) {
    xmm[1] = in[1];
    if (vlen * sizeof(T) > 2 * REGISTER_NUM_BYTES) {
    xmm[2] = in[2];
    if (vlen * sizeof(T) > 3 * REGISTER_NUM_BYTES) {
    xmm[3] = in[3];
    if (vlen * sizeof(T) > 4 * REGISTER_NUM_BYTES) {
      BAD;
    }}}}
  }

  // copy constructor:
  inline vec( const vec<T, vlen> & in) {
    xmm[0] = in.xmm[0];
    if (vlen * sizeof(T)>1* REGISTER_NUM_BYTES) {
    xmm[1] = in.xmm[1];
    if (vlen * sizeof(T)>2* REGISTER_NUM_BYTES) {
    xmm[2] = in.xmm[2];
    if (vlen * sizeof(T)>3* REGISTER_NUM_BYTES) {
    xmm[3] = in.xmm[3];
    if (vlen * sizeof(T)>4* REGISTER_NUM_BYTES) {
      BAD;
    }}}}
  }

  // assignment:
  inline vec& operator=(const vec & in) {
    xmm[0] = in.xmm[0];
    if (vlen * sizeof(T)>1* REGISTER_NUM_BYTES) {
    xmm[1] = in.xmm[1];
    if (vlen * sizeof(T)>2* REGISTER_NUM_BYTES) {
    xmm[2] = in.xmm[2];
    if (vlen * sizeof(T)>3* REGISTER_NUM_BYTES) {
    xmm[3] = in.xmm[3];
    if (vlen * sizeof(T)>4* REGISTER_NUM_BYTES) {
      BAD;
    }}}}
    return *this;
  }
