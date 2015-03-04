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
#ifndef __itkGPUImageToImageFilter_h
#define __itkGPUImageToImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkOpenCLKernelManager.h"

namespace itk
{
/** \class GPUImageToImageFilter
 *
 * \brief class to abstract the behaviour of the GPU filters.
 *
 * GPUImageToImageFilter is the GPU version of ImageToImageFilter.
 * This class can accept both CPU and GPU image as input and output,
 * and apply filter accordingly. If GPU is available for use, then
 * GPUGenerateData() is called. Otherwise, GenerateData() in the
 * parent class (i.e., ImageToImageFilter) will be called.
 *
 * \note This file was taken from ITK 4.1.0.
 * It was modified by Denis P. Shamonin and Marius Staring.
 * Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands.
 * Added functionality is described in the Insight Journal paper:
 * http://hdl.handle.net/10380/3393
 *
 * \ingroup ITKGPUCommon
 */
template< typename TInputImage, typename TOutputImage, typename TParentImageFilter
  = ImageToImageFilter< TInputImage, TOutputImage > >
class ITKOpenCL_EXPORT GPUImageToImageFilter : public TParentImageFilter
{
public:

  /** Standard class typedefs. */
  typedef GPUImageToImageFilter      Self;
  typedef TParentImageFilter         Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUImageToImageFilter, TParentImageFilter );

  /** Superclass typedefs. */
  typedef typename Superclass::DataObjectIdentifierType DataObjectIdentifierType;
  typedef typename Superclass::OutputImageRegionType    OutputImageRegionType;
  typedef typename Superclass::OutputImagePixelType     OutputImagePixelType;

  /** Some convenient typedefs. */
  typedef TInputImage                           InputImageType;
  typedef typename InputImageType::Pointer      InputImagePointer;
  typedef typename InputImageType::ConstPointer InputImageConstPointer;
  typedef typename InputImageType::RegionType   InputImageRegionType;
  typedef typename InputImageType::PixelType    InputImagePixelType;
  typedef TOutputImage                          OutputImageType;

  /** ImageDimension constants */
  itkStaticConstMacro( InputImageDimension, unsigned int, TInputImage::ImageDimension );
  itkStaticConstMacro( OutputImageDimension, unsigned int, TOutputImage::ImageDimension );

  // macro to set if GPU is used
  itkSetMacro( GPUEnabled, bool );
  itkGetConstMacro( GPUEnabled, bool );
  itkBooleanMacro( GPUEnabled );

  virtual void GraftOutput( DataObject * graft );

  virtual void GraftOutput( const DataObjectIdentifierType & key, DataObject * graft );

  virtual void SetNumberOfThreads( ThreadIdType _arg );

protected:

  GPUImageToImageFilter();
  ~GPUImageToImageFilter() {}

  virtual void PrintSelf( std::ostream & os, Indent indent ) const;

  virtual void GenerateData();

  virtual void GPUGenerateData() {}

  // GPU kernel manager
  typename OpenCLKernelManager::Pointer m_GPUKernelManager;

private:

  GPUImageToImageFilter( const Self & );   // purposely not implemented
  void operator=( const Self & );          // purposely not implemented

  bool m_GPUEnabled;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUImageToImageFilter.hxx"
#endif

#endif
