/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __itkGPUCastImageFilter_h
#define __itkGPUCastImageFilter_h

#include "itkCastImageFilter.h"
#include "itkSimpleDataObjectDecorator.h"

#include "itkGPUFunctorBase.h"
#include "itkOpenCLKernelManager.h"
#include "itkGPUUnaryFunctorImageFilter.h"

namespace itk
{
/** Create a helper GPU Kernel class for GPUCastImageFilter */
itkGPUKernelClassMacro( GPUCastImageFilterKernel );

/** \class GPUCastImageFilter
 * \brief GPU version of CastImageFilter.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
namespace Functor
{
template< typename TInput, typename TOutput >
class GPUCast : public GPUFunctorBase
{
public:

  GPUCast() {}

  ~GPUCast() {}

  /** Setup GPU kernel arguments for this functor.
   * Returns current argument index to set additional arguments in the GPU kernel.
   */
  int SetGPUKernelArguments( OpenCLKernelManager::Pointer KernelManager, int KernelHandle )
  {
    return 0;
  }


};

} // end of namespace Functor

/** GPUCastImageFilter class definition */
template< typename TInputImage, typename TOutputImage >
class ITK_EXPORT GPUCastImageFilter :
  public         GPUUnaryFunctorImageFilter<
  TInputImage, TOutputImage,
  Functor::GPUCast< typename TInputImage::PixelType,
  typename TOutputImage::PixelType >,
  CastImageFilter< TInputImage, TOutputImage > >
{
public:

  /** Standard class typedefs. */
  typedef GPUCastImageFilter Self;
  typedef GPUUnaryFunctorImageFilter<
    TInputImage, TOutputImage,
    Functor::GPUCast<
    typename TInputImage::PixelType,
    typename TOutputImage::PixelType >,
    CastImageFilter< TInputImage, TOutputImage > >   GPUSuperclass;
  typedef CastImageFilter< TInputImage, TOutputImage > CPUSuperclass;
  typedef SmartPointer< Self >                         Pointer;
  typedef SmartPointer< const Self >                   ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUCastImageFilter, GPUUnaryFunctorImageFilter );

  /** Pixel types. */
  typedef typename TInputImage::PixelType  InputPixelType;
  typedef typename TOutputImage::PixelType OutputPixelType;

  /** Type of DataObjects to use for scalar inputs */
  typedef SimpleDataObjectDecorator< InputPixelType > InputPixelObjectType;

protected:

  GPUCastImageFilter();
  virtual ~GPUCastImageFilter() {}

  /** Unlike CPU version, GPU version of binary threshold filter is not
  multi-threaded */
  virtual void GPUGenerateData( void );

private:

  GPUCastImageFilter( const Self & ); // purposely not implemented
  void operator=( const Self & );     // purposely not implemented

};

} // end of namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUCastImageFilter.hxx"
#endif

#endif /* __itkGPUCastImageFilter_h */
