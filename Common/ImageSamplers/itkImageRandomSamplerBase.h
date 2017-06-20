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
#ifndef __ImageRandomSamplerBase_h
#define __ImageRandomSamplerBase_h

#include "itkImageSamplerBase.h"

namespace itk
{

/** \class ImageRandomSamplerBase
 *
 * \brief This class is a base class for any image sampler that randomly picks samples.
 *
 * It adds the Set/GetNumberOfSamples function.
 *
 * \ingroup ImageSamplers
 */

template< class TInputImage >
class ImageRandomSamplerBase :
  public ImageSamplerBase< TInputImage >
{
public:

  /** Standard ITK-stuff. */
  typedef ImageRandomSamplerBase          Self;
  typedef ImageSamplerBase< TInputImage > Superclass;
  typedef SmartPointer< Self >            Pointer;
  typedef SmartPointer< const Self >      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( ImageRandomSamplerBase, ImageSamplerBase );

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass::DataObjectPointer            DataObjectPointer;
  typedef typename Superclass::OutputVectorContainerType    OutputVectorContainerType;
  typedef typename Superclass::OutputVectorContainerPointer OutputVectorContainerPointer;
  typedef typename Superclass::InputImageType               InputImageType;
  typedef typename Superclass::InputImagePointer            InputImagePointer;
  typedef typename Superclass::InputImageConstPointer       InputImageConstPointer;
  typedef typename Superclass::InputImageRegionType         InputImageRegionType;
  typedef typename Superclass::InputImagePixelType          InputImagePixelType;
  typedef typename Superclass::ImageSampleType              ImageSampleType;
  typedef typename Superclass::ImageSampleContainerType     ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer  ImageSampleContainerPointer;
  typedef typename Superclass::MaskType                     MaskType;

  /** The input image dimension. */
  itkStaticConstMacro( InputImageDimension, unsigned int,
    Superclass::InputImageDimension );

  /** Set the number of samples. */
  itkSetClampMacro( NumberOfSamples, unsigned long, 1, NumericTraits< unsigned long >::max() );

protected:

  /** The constructor. */
  ImageRandomSamplerBase();

  /** The destructor. */
  virtual ~ImageRandomSamplerBase() {}

  /** Multi-threaded function that does the work. */
  virtual void BeforeThreadedGenerateData( void );

  /** PrintSelf. */
  void PrintSelf( std::ostream & os, Indent indent ) const;

  /** Member variable used when threading. */
  std::vector< double > m_RandomNumberList;

private:

  /** The private constructor. */
  ImageRandomSamplerBase( const Self & );     // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );             // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageRandomSamplerBase.hxx"
#endif

#endif // end #ifndef __ImageRandomSamplerBase_h
