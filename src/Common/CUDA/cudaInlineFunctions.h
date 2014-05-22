/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __cudaInlineFunctions_h
#define __cudaInlineFunctions_h

#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include <assert.h>
#include "cudaMacro.h"

namespace cuda
{

#define cudaCheckMsg( msg ) __cudaCheckMsg( msg, __FILE__, __LINE__ )

inline void __cudaCheckMsg( const char *msg, const char *file, const int line )
{
  cudaError_t err = ::cudaGetLastError();
  if ( err != cudaSuccess )
  {
    const char* errcmsg = ::cudaGetErrorString( err );
    fprintf( stderr, "CUDA error: %s in file <%s>, line %i : %s.\n",
      msg, file, line, errcmsg );
    //assert( false );

    std::string errmsg = std::string( msg ) + ":: " + std::string( errcmsg );
    throw errmsg;
  }

#ifndef NDEBUG
  err = ::cudaThreadSynchronize();
  if ( err != cudaSuccess )
  {
    const char* errcmsg = ::cudaGetErrorString( err );
    fprintf( stderr, "cudaThreadSynchronize error: %s in file <%s>, line %i : %s.\n",
      msg, file, line, errcmsg );
    assert( false );

    std::string errmsg = std::string( msg ) + ":: " + std::string( errcmsg );
    throw errmsg;
  }
#endif /* NDEBUG */
}


template <class T>
inline T* cudaMalloc( size_t nof_elems )
{
  T* dst;
  size_t size = nof_elems * sizeof( T );
  ::cudaMalloc( (void**)&dst, size );
  cudaCheckMsg( "cudaMalloc failed!" );

  return dst;
}


template <class T>
inline T* cudaHostAlloc( size_t nof_elems, unsigned int flags = cudaHostAllocDefault )
{
  T* dst;
  size_t size = nof_elems * sizeof( T );
  ::cudaHostAlloc( (void**)&dst, size, flags );
  cudaCheckMsg( "cudaHostAlloc failed!" );

  return dst;
}


inline cudaError_t cudaMemcpy( void* dst, const void* src,
  size_t nof_elems, size_t sizeof_elem, cudaMemcpyKind direction )
{
  cudaError err = ::cudaMemcpy( dst, src, nof_elems * sizeof_elem, direction );
  cudaCheckMsg( "cudaMemcpy failed!" );
  return err;
}


template <class T>
inline void cudaMemcpy( T* dst, const T* src,
  size_t nof_elems, cudaMemcpyKind direction )
{
  size_t size = nof_elems * sizeof( T );
  ::cudaMemcpy( dst, src, size, direction );
  cudaCheckMsg( "cudaMemcpy failed!" );
}


template <class T>
inline void cudaMemset( T* dst, int value, size_t nof_elems )
{
  size_t size = nof_elems * sizeof( T );
  ::cudaMemset( dst, value, size );
  cudaCheckMsg( "cudaMemset failed!" );
}


template <typename T, typename Q>
inline cudaError_t cudaMemcpyToSymbol( const T& dst, const Q& src,
  cudaMemcpyKind direction )
{
  cudaError err = ::cudaMemcpyToSymbol( &dst, &src, sizeof(dst), 0, direction );
  cudaCheckMsg( "cudaMemcpyToSymbol failed!" );
  return err;
}


template <typename T>
inline cudaError_t cudaBindTextureToArray( const T& tex, cudaArray* array,
  const cudaChannelFormatDesc desc )
{
  cudaError_t err = ::cudaBindTextureToArray( &tex, array, &desc );
  cudaCheckMsg( "cudaBindTextureToArray failed!" );
  return err;
}


template <typename T>
inline cudaError_t cudaUnbindTexture( const T& tex )
{
  cudaError_t err = ::cudaUnbindTexture( &tex );
  cudaCheckMsg( "cudaUnbindTexture failed!" );
  return err;
}


/* Simple wrappers around functions we use to check return type.
 * In the future we might wrap the Driver-API around this so we can keep
 * using the high-level API.
 */
DBG_FUNC( cudaFreeArray, (struct cudaArray *array), (array) );
DBG_FUNC( cudaFree, (void *devPtr), (devPtr) );
DBG_FUNC( cudaMalloc3DArray, (struct cudaArray** arrayPtr,
  const struct cudaChannelFormatDesc* desc, struct cudaExtent extent),
  (arrayPtr, desc, extent) );
DBG_FUNC( cudaMemcpy3D, (const struct cudaMemcpy3DParms *p), (p) );
DBG_FUNC( cudaSetDevice, (int device), (device) );
DBG_FUNC( cudaGetDeviceProperties, (struct cudaDeviceProp *prop, int device),
  (prop, device) );

}; /* cuda */

#endif // end #ifndef __cudaInlineFunctions_h
