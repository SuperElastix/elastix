/* This file provides the implementation of the  vec< int32 , 4>  type
 * Only for including in emm_vec.hxx.
 * Don't include anywhere else.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 19-4-2012
 */



#ifdef INCLUDE_SSE4
// other functions (min/max ..) (int32, length 4)
inline vec<int32_t, 4> max(const vec<int32_t, 4> &a, const vec<int32_t, 4> &b){
  return vec<int32_t,4>(_mm_max_epi32(a.xmmi[0], b.xmmi[0]));
}
inline vec<int32_t, 4> min(const vec<int32_t, 4> &a, const vec<int32_t, 4> &b){
  return vec<int32_t,4>(_mm_min_epi32(a.xmmi[0], b.xmmi[0]));
}
#endif
