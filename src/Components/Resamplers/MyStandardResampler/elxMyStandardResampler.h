/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxMyStandardResampler_h
#define __elxMyStandardResampler_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkResampleImageFilter.h"

namespace elastix
{

  /**
   * \class MyStandardResampler
   * \brief A resampler based on the itk::ResampleImageFilter.
   *
   * The parameters used in this class are:
   * \parameter Resampler: Select this resampler as follows:\n
   *    <tt>(Resampler "DefaultResampler")</tt>
   *
   * \ingroup Resamplers
   */

  template < class TElastix >
    class MyStandardResampler :
      public ResamplerBase<TElastix>::ITKBaseType,
      public ResamplerBase<TElastix>
  {
  public:

    /** Standard ITK-stuff. */
    typedef MyStandardResampler                             Self;
    typedef typename ResamplerBase<TElastix>::ITKBaseType   Superclass1;
    typedef ResamplerBase<TElastix>                         Superclass2;
    typedef itk::SmartPointer<Self>                         Pointer;
    typedef itk::SmartPointer<const Self>                   ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro( MyStandardResampler, ResampleImageFilter );

    /** Name of this class.
     * Use this name in the parameter file to select this specific resampler. \n
     * example: <tt>(Resampler "DefaultResampler")</tt>\n
     */
    elxClassNameMacro( "DefaultResampler" );

    /** Typedef's inherited from the superclass. */
    typedef typename Superclass1::InputImageType            InputImageType;
    typedef typename Superclass1::OutputImageType           OutputImageType;
    typedef typename Superclass1::InputImagePointer         InputImagePointer;
    typedef typename Superclass1::OutputImagePointer        OutputImagePointer;
    typedef typename Superclass1::InputImageRegionType      InputImageRegionType;
    typedef typename Superclass1::TransformType             TransformType;
    typedef typename Superclass1::TransformPointerType      TransformPointerType;
    typedef typename Superclass1::InterpolatorType          InterpolatorType;
    typedef typename Superclass1::InterpolatorPointerType   InterpolatePointerType;
    typedef typename Superclass1::SizeType                  SizeType;
    typedef typename Superclass1::IndexType                 IndexType;
    typedef typename Superclass1::PointType                 PointType;
    typedef typename Superclass1::PixelType                 PixelType;
    typedef typename Superclass1::OutputImageRegionType     OutputImageRegionType;
    typedef typename Superclass1::SpacingType               SpacingType;
    typedef typename Superclass1::OriginPointType           OriginPointType;

    /** Typedef's from the ResamplerBase. */
    typedef typename Superclass2::ElastixType           ElastixType;
    typedef typename Superclass2::ElastixPointer        ElastixPointer;
    typedef typename Superclass2::ConfigurationType     ConfigurationType;
    typedef typename Superclass2::ConfigurationPointer  ConfigurationPointer;
    typedef typename Superclass2::RegistrationType      RegistrationType;
    typedef typename Superclass2::RegistrationPointer   RegistrationPointer;
    typedef typename Superclass2::ITKBaseType           ITKBaseType;

    /* Nothing to add. In the baseclass already everything is done what should be done. */

  protected:

    /** The constructor. */
    MyStandardResampler() {}
    /** The destructor. */
    virtual ~MyStandardResampler() {}

  private:

    /** The private constructor. */
    MyStandardResampler( const Self& ); // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );      // purposely not implemented

  }; // end class MyStandardResampler


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMyStandardResampler.hxx"
#endif

#endif // end #ifndef __elxMyStandardResampler_h
