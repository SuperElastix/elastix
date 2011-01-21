/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __ReducedDimensionImageGridSampler_h
#define __ReducedDimensionImageGridSampler_h

#include "itkImageGridSampler.h"

namespace itk
{

  /** \class ReducedDimensionImageGridSampler
   *
   * \brief Samples image voxels on a regular grid within a subregion
   *    defined by a fixed dimension index
   *
   * This ImageSampler samples voxels that lie on a regular grid.
   * The grid can be specified by an integer downsampling factor for
   * each dimension.
   *
   * \parameter ReducedDimension: This parameter determines the dimension
   *    to keep fixed during sampling.
   * \parameter ReducedDimensionIndex: This parameter determines the index
   *    number of ReducedDimension to keep fixed during sampling.
   */

  template < class TInputImage >
  class ReducedDimensionImageGridSampler :
    public ImageGridSampler< TInputImage >
  {
  public:

    /** Standard ITK-stuff. */
    typedef ReducedDimensionImageGridSampler  Self;
    typedef ImageGridSampler< TInputImage >   Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( ReducedDimensionImageGridSampler, ImageGridSampler );

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

    /** The input image dimension. */
    itkStaticConstMacro( InputImageDimension, unsigned int,
      Superclass::InputImageDimension );

    /** Other typdefs. */
    typedef typename InputImageType::IndexType    InputImageIndexType;
    typedef typename InputImageType::PointType    InputImagePointType;

    /** Typedefs for support of user defined grid spacing for the spatial samples. */
    typedef typename Superclass::SampleGridSpacingType      SampleGridSpacingType;
    typedef typename Superclass::SampleGridSpacingValueType SampleGridSpacingValueType;
    typedef typename Superclass::SampleGridSizeType         SampleGridSizeType;
    typedef typename Superclass::SampleGridIndexType        SampleGridIndexType;
    typedef typename Superclass::InputImageSizeType         InputImageSizeType;

    /** Set and get macro for fixed dimension. */
    itkSetMacro( ReducedDimension, unsigned int );
    itkSetMacro( ReducedDimensionIndex, unsigned int );
    itkGetConstMacro( ReducedDimension, unsigned int );
    itkGetConstMacro( ReducedDimensionIndex, unsigned int );

  protected:

    /** The constructor. */
    ReducedDimensionImageGridSampler()
    {
      this->m_RequestedNumberOfSamples = 0;
    }

    /** The destructor. */
    virtual ~ReducedDimensionImageGridSampler() {};

    /** PrintSelf. */
    void PrintSelf( std::ostream& os, Indent indent ) const;

    /** Set input image region. */
    void SetInputImageRegion( const InputImageRegionType _arg, unsigned int pos );

    /** The dimension to keep fixed during sampling. */
    unsigned int m_ReducedDimension;
    unsigned int m_ReducedDimensionIndex;

  private:

    /** The private constructor. */
    ReducedDimensionImageGridSampler( const Self& );  // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );               // purposely not implemented

  }; // end class ReducedDimensionImageGridSampler

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkReducedDimensionImageGridSampler.txx"
#endif

#endif // end #ifndef __ReducedDimensionImageGridSampler_h
