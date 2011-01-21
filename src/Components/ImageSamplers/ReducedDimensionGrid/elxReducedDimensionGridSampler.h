/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxReducedDimensionGridSampler_h
#define __elxReducedDimensionGridSampler_h

#include "itkReducedDimensionImageGridSampler.h"
#include "elxIncludes.h"

namespace elastix
{

using namespace itk;

  /**
   * \class ReducedDimensionGridSampler
   * \brief An interpolator based on the itk::ReducedDimensionImageGridSampler.
   *
   * This image sampler samples voxels on a uniform grid, but ignores the
   * user-defined dimension during sampling and sets it to the given index.
   * This sampler is in most cases not recommended.
   *
   * This sampler does not react on the
   * NewSamplesEveryIteration parameter.
   *
   * The parameters used in this class are:
   * \parameter ReducedDimension: This parameter determines the dimension
   *    to keep fixed during sampling.
   * \parameter ReducedDimensionIndex: This parameter determines the index
   *    number of ReducedDimension to keep fixed during sampling.
   *
   * \ingroup ImageSamplers
   */

  template < class TElastix >
    class ReducedDimensionGridSampler :
    public
      ReducedDimensionImageGridSampler<
      ITK_TYPENAME elx::ImageSamplerBase<TElastix>::InputImageType >,
    public
      elx::ImageSamplerBase<TElastix>
  {
  public:

    /** Standard ITK-stuff. */
    typedef ReducedDimensionGridSampler         Self;
    typedef ReducedDimensionImageGridSampler<
      typename elx::ImageSamplerBase<TElastix>::InputImageType >  Superclass1;
    typedef elx::ImageSamplerBase<TElastix>     Superclass2;
    typedef SmartPointer<Self>                  Pointer;
    typedef SmartPointer<const Self>            ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro( ReducedDimensionGridSampler, ReducedDimensionImageGridSampler );

    /** Name of this class.
     * Use this name in the parameter file to select this specific interpolator. \n
     * example: <tt>(ImageSampler "ReducedDimensionGrid")</tt>\n
     */
    elxClassNameMacro( "ReducedDimensionGrid" );

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
    typedef typename Superclass1::SampleGridSpacingType        GridSpacingType;
    typedef typename Superclass1::SampleGridSpacingValueType   SampleGridSpacingValueType;

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

    /** Execute stuff before each resolution:
     * \li Set the sampling grid size.
     */
    virtual void BeforeEachResolution(void);

    /** Execute stuff before registration:
    * \li Set the dimension to keep fixed.
    */
    virtual void BeforeRegistration(void);

  protected:

    /** The constructor. */
    ReducedDimensionGridSampler() {}
    /** The destructor. */
    virtual ~ReducedDimensionGridSampler() {}

  private:

    /** The private constructor. */
    ReducedDimensionGridSampler( const Self& ); // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );      // purposely not implemented

  }; // end class GridSampler


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxReducedDimensionGridSampler.hxx"
#endif

#endif // end #ifndef __elxReducedDimensionGridSampler_h

