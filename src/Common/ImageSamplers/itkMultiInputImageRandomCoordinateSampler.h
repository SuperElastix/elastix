/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __MultiInputImageRandomCoordinateSampler_h
#define __MultiInputImageRandomCoordinateSampler_h

#include "itkImageRandomSamplerBase.h"
#include "itkInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"


namespace itk
{

  /** \class MultiInputImageRandomCoordinateSampler
   *
   * \brief Samples an image by randomly composing a set of physical coordinates
   *
   * This image sampler generates not only samples that correspond with
   * pixel locations, but selects points in physical space.
	 *
	 * \ingroup ImageSamplers
   */

  template < class TInputImage >
  class MultiInputImageRandomCoordinateSampler :
    public ImageRandomSamplerBase< TInputImage >
  {
  public:

    /** Standard ITK-stuff. */
    typedef MultiInputImageRandomCoordinateSampler                Self;
    typedef ImageRandomSamplerBase< TInputImage >       Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( MultiInputImageRandomCoordinateSampler, ImageRandomSamplerBase );

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
    typedef typename Superclass::MaskType                     MaskType;
    typedef typename Superclass::InputImageSizeType           InputImageSizeType;
    typedef typename InputImageType::SpacingType              InputImageSpacingType;
    typedef typename Superclass::InputImageIndexType          InputImageIndexType;
    typedef typename Superclass::InputImagePointType          InputImagePointType;
    typedef typename Superclass::InputImagePointValueType     InputImagePointValueType;
    typedef typename Superclass::ImageSampleValueType         ImageSampleValueType;

    /** The input image dimension. */
    itkStaticConstMacro( InputImageDimension, unsigned int,
      Superclass::InputImageDimension );

    /** This image sampler samples the image on physical coordinates and thus
     * needs an interpolator.
     */
    typedef double                                              CoordRepType;
    typedef InterpolateImageFunction<
      InputImageType, CoordRepType >                            InterpolatorType;
    typedef BSplineInterpolateImageFunction<
      InputImageType, CoordRepType, double>                     DefaultInterpolatorType;

    /** The random number generator used to generate random coordinates. */
    typedef itk::Statistics::MersenneTwisterRandomVariateGenerator RandomGeneratorType;

    /** Set/Get the interpolator. A 3rd order BSpline interpolator is used by default. */
    itkSetObjectMacro( Interpolator, InterpolatorType );
    itkGetObjectMacro( Interpolator, InterpolatorType );

    /** Set/Get the sample region size (in mm). Only needed when UseRandomSampleRegion==true;
     * default: filled with ones.
     */
    itkSetMacro( SampleRegionSize, InputImageSpacingType );
    itkGetConstReferenceMacro( SampleRegionSize, InputImageSpacingType );

    /** Set/Get whether to use randomly selected sample regions, or just the whole image
     * Default: false. */
    itkGetConstMacro( UseRandomSampleRegion, bool );
    itkSetMacro( UseRandomSampleRegion, bool );

  protected:

    typedef typename InterpolatorType::ContinuousIndexType   InputImageContinuousIndexType;

    /** The constructor. */
    MultiInputImageRandomCoordinateSampler();

    /** The destructor. */
    virtual ~MultiInputImageRandomCoordinateSampler() {};

    /** PrintSelf. */
    void PrintSelf( std::ostream& os, Indent indent ) const;

    /** Function that does the work. */
    virtual void GenerateData( void );

    /** Generate a point randomly in a bounding box.
     * This method can be overwritten in subclasses if a different distribution is desired. */
    virtual void GenerateRandomCoordinate(
      const InputImageContinuousIndexType & smallestContIndex,
      const InputImageContinuousIndexType & largestContIndex,
      InputImageContinuousIndexType &       randomContIndex);

    typename InterpolatorType::Pointer    m_Interpolator;
    typename RandomGeneratorType::Pointer m_RandomGenerator;
    InputImageSpacingType                 m_SampleRegionSize;

    /** Generate the two corners of a sampling region. */
    virtual void GenerateSampleRegion(
      InputImageContinuousIndexType & smallestContIndex,
      InputImageContinuousIndexType & largestContIndex );

  private:

    /** The private constructor. */
    MultiInputImageRandomCoordinateSampler( const Self& );          // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );            // purposely not implemented

    bool          m_UseRandomSampleRegion;

  }; // end class MultiInputImageRandomCoordinateSampler


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiInputImageRandomCoordinateSampler.txx"
#endif

#endif // end #ifndef __MultiInputImageRandomCoordinateSampler_h

