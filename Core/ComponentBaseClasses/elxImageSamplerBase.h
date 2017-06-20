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

#ifndef __elxImageSamplerBase_h
#define __elxImageSamplerBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"

#include "itkImageSamplerBase.h"

namespace elastix
{
//using namespace itk; not here because itk::ImageSamplerBase exists also.

/**
 * \class ImageSamplerBase
 * \brief This class is the elastix base class for all ImageSamplers.
 *
 * This class contains all the common functionality for ImageSamplers.
 *
 * \ingroup ImageSamplers
 * \ingroup ComponentBaseClasses
 */

template< class TElastix >
class ImageSamplerBase : public BaseComponentSE< TElastix >
{
public:

  /** Standard ITK-stuff. */
  typedef ImageSamplerBase            Self;
  typedef BaseComponentSE< TElastix > Superclass;

  /** Run-time type information (and related methods). */
  itkTypeMacro( ImageSamplerBase, BaseComponentSE );

  /** Typedefs inherited from Elastix. */
  typedef typename Superclass::ElastixType          ElastixType;
  typedef typename Superclass::ElastixPointer       ElastixPointer;
  typedef typename Superclass::ConfigurationType    ConfigurationType;
  typedef typename Superclass::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass::RegistrationType     RegistrationType;
  typedef typename Superclass::RegistrationPointer  RegistrationPointer;

  /** Other typedef's. */
  typedef typename ElastixType::FixedImageType InputImageType;

  /** ITKBaseType. */
  typedef itk::ImageSamplerBase< InputImageType > ITKBaseType;

  /** Cast to ITKBaseType. */
  virtual ITKBaseType * GetAsITKBaseType( void )
  {
    return dynamic_cast< ITKBaseType * >( this );
  }


  /** Cast to ITKBaseType, to use in const functions. */
  virtual const ITKBaseType * GetAsITKBaseType( void ) const
  {
    return dynamic_cast< const ITKBaseType * >( this );
  }


  /** Execute stuff before each resolution:
   * \li Give a warning when NewSamplesEveryIteration is specified,
   * but the sampler is ignoring it.
   */
  virtual void BeforeEachResolutionBase( void );

protected:

  /** The constructor. */
  ImageSamplerBase() {}
  /** The destructor. */
  virtual ~ImageSamplerBase() {}

private:

  /** The private constructor. */
  ImageSamplerBase( const Self & );   // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );     // purposely not implemented

};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxImageSamplerBase.hxx"
#endif

#endif // end #ifndef __elxImageSamplerBase_h
