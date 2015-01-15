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
#ifndef __elxCUDAResampler_h
#define __elxCUDAResampler_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkCUDAResampleImageFilter.h"

namespace elastix
{

/**
 * \class CUDAResampler
 * \brief A resampler based on the itk::CUDAResampleImageFilter.
 *
 * \warning The implementation is currently very limited: only
 * a single third order B-spline transform is supported for 3D
 * images together with third order B-spline interpolation.
 *
 * The parameters used in this class are:
 * \parameter Resampler: Select this resampler as follows:\n
 *    <tt>(Resampler "CUDAResampler")</tt>
 *
 * \ingroup Resamplers
 */

template< class TElastix >
class CUDAResampler :
  public itk::itkCUDAResampleImageFilter<
  typename ResamplerBase< TElastix >::InputImageType,
  typename ResamplerBase< TElastix >::OutputImageType,
  typename ResamplerBase< TElastix >::CoordRepType >,
  public ResamplerBase< TElastix >
{
public:

  /** Standard ITK-stuff. */
  typedef CUDAResampler Self;
  typedef itk::itkCUDAResampleImageFilter<
    typename ResamplerBase< TElastix >::InputImageType,
    typename ResamplerBase< TElastix >::OutputImageType,
    typename ResamplerBase< TElastix >::CoordRepType >      Superclass1;
  typedef ResamplerBase< TElastix >       Superclass2;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( CUDAResampler, itk::itkCUDAResampleImageFilter );

  /** Name of this class.
   * Use this name in the parameter file to select this specific resampler. \n
   * example: <tt>(Resampler "CUDAResampler")</tt>\n
   */
  elxClassNameMacro( "CUDAResampler" );

  /** Typedef's inherited from the superclass. */
  typedef typename Superclass1::InputImageType          InputImageType;
  typedef typename Superclass1::OutputImageType         OutputImageType;
  typedef typename Superclass1::InputImagePointer       InputImagePointer;
  typedef typename Superclass1::OutputImagePointer      OutputImagePointer;
  typedef typename Superclass1::InputImageRegionType    InputImageRegionType;
  typedef typename Superclass1::TransformType           TransformType;
  typedef typename Superclass1::TransformPointerType    TransformPointerType;
  typedef typename Superclass1::InterpolatorType        InterpolatorType;
  typedef typename Superclass1::InterpolatorPointerType InterpolatePointerType;
  typedef typename Superclass1::SizeType                SizeType;
  typedef typename Superclass1::IndexType               IndexType;
  typedef typename Superclass1::PointType               PointType;
  typedef typename Superclass1::PixelType               PixelType;
  typedef typename Superclass1::OutputImageRegionType   OutputImageRegionType;
  typedef typename Superclass1::SpacingType             SpacingType;
  typedef typename Superclass1::OriginPointType         OriginPointType;
  typedef typename Superclass1::ValidTransformPointer   ValidTransformPointer;

  /** Typedef's from the ResamplerBase. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

  /* . */
  virtual int BeforeAll( void );

  virtual void BeforeRegistration( void );

  /** Function to read parameters from a file. */
  virtual void ReadFromFile( void );

  /** Function to write parameters to a file. */
  virtual void WriteToFile( void ) const;

protected:

  /** The constructor. */
  CUDAResampler() {}
  /** The destructor. */
  virtual ~CUDAResampler() {}

  /** Overwrite from itkCUDAResampleImageFilter.
   * We simply call the Superclass and print the warning messages to elxout.
   */
  virtual void CheckForValidConfiguration( ValidTransformPointer & bSplineTransform );

private:

  /** The private constructor. */
  CUDAResampler( const Self & );   // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );  // purposely not implemented

};

// end class CUDAResampler

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxCUDAResampler.hxx"
#endif

#endif // end #ifndef __elxCUDAResampler_h
