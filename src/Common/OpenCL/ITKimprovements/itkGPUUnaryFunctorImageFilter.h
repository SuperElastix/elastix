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
/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __itkGPUUnaryFunctorImageFilter_h
#define __itkGPUUnaryFunctorImageFilter_h

#include "itkGPUFunctorBase.h"
#include "itkGPUInPlaceImageFilter.h"
#include "itkUnaryFunctorImageFilter.h"

namespace itk
{
/** \class GPUUnaryFunctorImageFilter
 * \brief Implements pixel-wise generic operation on one image using the GPU.
 *
 * GPU version of unary functor image filter.
 * GPU Functor handles parameter setup for the GPU kernel.
 *
 * \note This file was taken from ITK 4.1.0.
 * It was modified by Denis P. Shamonin and Marius Staring.
 * Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands.
 * Added functionality is described in the Insight Journal paper:
 * http://hdl.handle.net/10380/3393
 *
 * \ingroup   ITKGPUCommon
 */
template< typename TInputImage, typename TOutputImage, typename TFunction, typename TParentImageFilter
  = InPlaceImageFilter< TInputImage, TOutputImage > >
class ITKOpenCL_EXPORT GPUUnaryFunctorImageFilter : public GPUInPlaceImageFilter< TInputImage, TOutputImage,
  TParentImageFilter >
{
public:

  /** Standard class typedefs. */
  typedef GPUUnaryFunctorImageFilter                         Self;
  typedef TParentImageFilter                                 CPUSuperclass;
  typedef GPUInPlaceImageFilter< TInputImage, TOutputImage > GPUSuperclass;
  typedef SmartPointer< Self >                               Pointer;
  typedef SmartPointer< const Self >                         ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUUnaryFunctorImageFilter, GPUInPlaceImageFilter );

  /** Some typedefs. */
  typedef TFunction FunctorType;

  typedef TInputImage                              InputImageType;
  typedef typename    InputImageType::ConstPointer InputImagePointer;
  typedef typename    InputImageType::RegionType   InputImageRegionType;
  typedef typename    InputImageType::PixelType    InputImagePixelType;

  typedef TOutputImage                             OutputImageType;
  typedef typename     OutputImageType::Pointer    OutputImagePointer;
  typedef typename     OutputImageType::RegionType OutputImageRegionType;
  typedef typename     OutputImageType::PixelType  OutputImagePixelType;

  /** ImageDimension constants */
  itkStaticConstMacro( InputImageDimension, unsigned int,
    TInputImage::ImageDimension );
  itkStaticConstMacro( OutputImageDimension, unsigned int,
    TOutputImage::ImageDimension );

  FunctorType & GetFunctor()
  {
    return m_Functor;
  }


  const FunctorType & GetFunctor() const
  {
    return m_Functor;
  }


  /** Set the functor object. */
  void SetFunctor( const FunctorType & functor )
  {
    if( m_Functor != functor )
    {
      m_Functor = functor;
      this->Modified();
    }
  }


protected:

  GPUUnaryFunctorImageFilter() {}

  virtual ~GPUUnaryFunctorImageFilter() {}

  virtual void GenerateOutputInformation();

  virtual void GPUGenerateData();

  /** GPU kernel handle is defined here instead of in the child class
   * because GPUGenerateData() in this base class is used. */
  int m_UnaryFunctorImageFilterGPUKernelHandle;

private:

  GPUUnaryFunctorImageFilter( const Self & );   // purposely not implemented
  void operator=( const Self & );               // purposely not implemented

  FunctorType m_Functor;
};

} // end of namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUUnaryFunctorImageFilter.hxx"
#endif

#endif
