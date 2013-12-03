/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxFullSampler_h
#define __elxFullSampler_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkImageFullSampler.h"

namespace elastix
{


  /**
   * \class FullSampler
   * \brief An interpolator based on the itk::ImageFullSampler.
   *
   * This image sampler samples all voxels in
   * the InputImageRegion.
   *
   * This sampler does not react to the NewSamplesEveryIteration parameter.
   *
   * The parameters used in this class are:
   * \parameter ImageSampler: Select this image sampler as follows:\n
   *    <tt>(ImageSampler "Full")</tt>
   *
   * \ingroup ImageSamplers
   */

  template < class TElastix >
    class FullSampler :
    public
      itk::ImageFullSampler<
      typename elx::ImageSamplerBase<TElastix>::InputImageType >,
    public
      elx::ImageSamplerBase<TElastix>
  {
  public:

    /** Standard ITK-stuff. */
    typedef FullSampler                              Self;
    typedef itk::ImageFullSampler<
      typename elx::ImageSamplerBase<TElastix>::InputImageType >
                                                     Superclass1;
    typedef elx::ImageSamplerBase<TElastix>          Superclass2;
    typedef itk::SmartPointer<Self>                  Pointer;
    typedef itk::SmartPointer<const Self>            ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro( FullSampler, itk::ImageFullSampler );

    /** Name of this class.
     * Use this name in the parameter file to select this specific interpolator. \n
     * example: <tt>(ImageSampler "Full")</tt>\n
     */
    elxClassNameMacro( "Full" );

    /** Typedefs inherited from the superclass. */
    typedef typename Superclass1::DataObjectPointer            DataObjectPointer;
    typedef typename Superclass1::OutputVectorContainerType    OutputVectorContainerType;
    typedef typename Superclass1::OutputVectorContainerPointer OutputVectorContainerPointer;
    typedef typename Superclass1::InputImageType               InputImageType;
    typedef typename Superclass1::InputImagePointer            InputImagePointer;
    typedef typename Superclass1::InputImageConstPointer       InputImageConstPointer;
    typedef typename Superclass1::InputImageRegionType         InputImageRegionType;
    typedef typename Superclass1::InputImagePixelType          InputImagePixelType;
    typedef typename Superclass1::ImageSampleType              ImageSampleType;
    typedef typename Superclass1::ImageSampleContainerType     ImageSampleContainerType;
    typedef typename Superclass1::MaskType                     MaskType;
    typedef typename Superclass1::InputImageIndexType          InputImageIndexType;
    typedef typename Superclass1::InputImagePointType          InputImagePointType;

    /** The input image dimension. */
    itkStaticConstMacro( InputImageDimension, unsigned int, Superclass1::InputImageDimension );

    /** Typedefs inherited from Elastix. */
    typedef typename Superclass2::ElastixType               ElastixType;
    typedef typename Superclass2::ElastixPointer            ElastixPointer;
    typedef typename Superclass2::ConfigurationType         ConfigurationType;
    typedef typename Superclass2::ConfigurationPointer      ConfigurationPointer;
    typedef typename Superclass2::RegistrationType          RegistrationType;
    typedef typename Superclass2::RegistrationPointer       RegistrationPointer;
    typedef typename Superclass2::ITKBaseType               ITKBaseType;

  protected:

    /** The constructor. */
    FullSampler() {}
    /** The destructor. */
    virtual ~FullSampler() {}

  private:

    /** The private constructor. */
    FullSampler( const Self& ); // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );      // purposely not implemented

  };


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxFullSampler.hxx"
#endif

#endif // end #ifndef __elxFullSampler_h

