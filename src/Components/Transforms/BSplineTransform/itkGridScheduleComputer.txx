/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkGridScheduleComputer_TXX__
#define __itkGridScheduleComputer_TXX__

#include "itkGridScheduleComputer.h"

#include "itkImageRegionExclusionConstIteratorWithIndex.h"


namespace itk
{

/**
 * ********************* Constructor ****************************
 */
  
template < typename TTransformScalarType, unsigned int VImageDimension >
GridScheduleComputer<TTransformScalarType, VImageDimension>
::GridScheduleComputer()
{
  this->m_BSplineOrder = 3;
  this->m_InitialTransform = 0;
  this->SetDefaultSchedule( 3, 2.0 );

} // end Constructor()


/**
 * ********************* SetDefaultGridSpacingSchedule ****************************
 */

template < typename TTransformScalarType, unsigned int VImageDimension >
void
GridScheduleComputer<TTransformScalarType, VImageDimension>
::SetDefaultSchedule( unsigned int levels, double upsamplingFactor )
{
  /** Set member variables. */
  this->m_NumberOfLevels = levels;
  this->SetUpsamplingFactor( upsamplingFactor );

  /** Initialize the schedule. */
  GridSpacingFactorType factors;
  factors.Fill( 1.0 );
  this->m_GridSpacingFactors.clear();
  this->m_GridSpacingFactors.resize( levels, factors );

  /** Setup a default schedule. */
  float factor = this->m_UpsamplingFactor;
  for ( int i = levels - 2; i > -1; --i )
  {
    this->m_GridSpacingFactors[ i ] *= factor;
    factor *= this->m_UpsamplingFactor;
  }

} // end SetDefaultGridSpacingSchedule()


/**
 * ********************* SetGridSpacingSchedule ****************************
 */

template < typename TTransformScalarType, unsigned int VImageDimension >
void
GridScheduleComputer<TTransformScalarType, VImageDimension>
::SetSchedule( const VectorGridSpacingFactorType & schedule )
{
  /** Set member variables. */
  this->m_GridSpacingFactors = schedule;
  this->m_NumberOfLevels = schedule.size();

} // end SetGridSpacingSchedule()


/**
 * ********************* GetGridSpacingSchedule ****************************
 */

template < typename TTransformScalarType, unsigned int VImageDimension >
void
GridScheduleComputer<TTransformScalarType, VImageDimension>
::GetSchedule( VectorGridSpacingFactorType & schedule ) const
{
  schedule = this->m_GridSpacingFactors;

} // end GetGridSpacingSchedule()


/**
 * ********************* ComputeBSplineGrid ****************************
 */

template < typename TTransformScalarType, unsigned int VImageDimension >
void
GridScheduleComputer<TTransformScalarType, VImageDimension>
::ComputeBSplineGrid( void )
{
  OriginType imageOrigin;
  SpacingType imageSpacing, finalGridSpacing;

  /** Apply the initial transform. */    
  this->ApplyInitialTransform( imageOrigin, imageSpacing, finalGridSpacing );

  /** Set the appropriate sizes. */
  this->m_GridOrigins.resize( this->m_NumberOfLevels );
  this->m_GridRegions.resize( this->m_NumberOfLevels );
  this->m_GridSpacings.resize( this->m_NumberOfLevels );

  /** For all levels ... */
  for ( unsigned int res = 0; res < this->m_NumberOfLevels; ++res )
  {
    /** For all dimensions ... */
    SizeType size = this->m_ImageRegion.GetSize();
    SizeType gridsize;
    for ( unsigned int dim = 0; dim < Dimension; ++dim )
    {
      /** Compute the grid spacings. */
      double gridSpacing
        = finalGridSpacing[ dim ] * this->m_GridSpacingFactors[ res ][ dim ];
      this->m_GridSpacings[ res ][ dim ] = gridSpacing;

      /** Compute the grid size without the extra grid points at the edges. */
      const unsigned int bareGridSize = static_cast<unsigned int>(
        vcl_ceil( size[ dim ] * imageSpacing[ dim ] / gridSpacing ) );

      /** The number of B-spline grid nodes is the bareGridSize plus the
       * B-spline order more grid nodes.
       */
      gridsize[ dim ] = static_cast<SizeValueType>(
        bareGridSize + this->m_BSplineOrder );

      /** Compute the origin of the B-spline grid. */
      this->m_GridOrigins[ res ][ dim ] = imageOrigin[ dim ] -
        ( ( gridsize[ dim ] - 1 ) * gridSpacing
        - ( size[ dim ] - 1 ) * imageSpacing[ dim ] ) / 2.0;
    }

    /** Set the grid region. */
    this->m_GridRegions[ res ].SetSize( gridsize );
  }

} // end ComputeBSplineGrid()


/**
 * ********************* ApplyInitialTransform ****************************
 *
 * This function adapts the m_ImageOrigin and m_ImageSpacing.
 * This makes sure that the BSpline grid is located at the position
 * of the fixed image after undergoing the initial transform.
 */

template < typename TTransformScalarType, unsigned int VImageDimension >
void
GridScheduleComputer<TTransformScalarType, VImageDimension>
::ApplyInitialTransform(
  OriginType & imageOrigin,
  SpacingType & imageSpacing,
  SpacingType & finalGridSpacing ) const
{
  /** Check for the existence of an initial transform. */
  if ( this->m_InitialTransform.IsNull() )
  {
    imageOrigin = this->m_ImageOrigin;
    imageSpacing = this->m_ImageSpacing;
    finalGridSpacing = this->m_FinalGridSpacing;
    return;
  }

  /** We have to determine a bounding box around the fixed image after
   * applying the initial transform. This is done by iterating over the
   * the boundary of the fixed image, evaluating the initial transform
   * at those points, and keeping track of the minimum/maximum transformed
   * coordinate in each dimension.
   */

  /** Create a temporary image. As small as possible, for memory savings. */
  typedef Image< unsigned char, Dimension >     ImageType;//bool??
  typename ImageType::Pointer image = ImageType::New();
  image->SetOrigin( this->m_ImageOrigin );
  image->SetSpacing( this->m_ImageSpacing );
  image->SetRegions( this->m_ImageRegion );
  image->Allocate();

  /** The points that define the bounding box. */
  OriginType maxPoint;
  OriginType minPoint;
  maxPoint.Fill( NumericTraits< TransformScalarType >::NonpositiveMin() );
  minPoint.Fill( NumericTraits< TransformScalarType >::max() );

  /** An iterator over the boundary of the image. */
  typedef ImageRegionExclusionConstIteratorWithIndex<
    ImageType >                               BoundaryIteratorType;
  BoundaryIteratorType bit( image, this->m_ImageRegion );
  bit.SetExclusionRegionToInsetRegion();
  bit.GoToBegin();

  /** Start loop over boundary; determines minPoint and maxPoint. */
  typedef typename ImageType::IndexType IndexType;
  while ( !bit.IsAtEnd() )
  {
    /** Get index, transform to physical point, apply initial transform.
     * NB: the OutputPointType of the initial transform by definition equals
     * the InputPointType of this transform.
     */
    IndexType inputIndex = bit.GetIndex();
    OriginType inputPoint;
    image->TransformIndexToPhysicalPoint( inputIndex, inputPoint );
    typename TransformType::OutputPointType outputPoint = 
      this->m_InitialTransform->TransformPoint( inputPoint );

    /** Update minPoint and maxPoint. */
    for ( unsigned int i = 0; i < Dimension; i++ )
    {
      TransformScalarType & outi = outputPoint[ i ];
      TransformScalarType & maxi = maxPoint[ i ];
      TransformScalarType & mini = minPoint[ i ];
      if ( outi > maxi )
      {
        maxi = outi;
      }
      if ( outi < mini )
      {
        mini = outi;
      }
    }

    /** Step to next voxel. */
    ++bit;

  } // end while loop over image boundary

  /** Set minPoint as the new "ImageOrigin" (between quotes, since it
   * is not really the origin of the fixedImage anymore).
   */
  imageOrigin = minPoint;

  /** Compute the new "ImageSpacing" in each dimension. */
  const double smallnumber = NumericTraits<double>::epsilon();
  for ( unsigned int i = 0; i < Dimension; i++ )
  {
    /** Compute the length of the fixed image (in mm) for dimension i. */
    double oldLength_i = 
      this->m_ImageSpacing[ i ] * static_cast<double>(
      this->m_ImageRegion.GetSize()[ i ] - 1 );

    /** Compute the length of the bounding box (in mm) for dimension i. */
    double newLength_i = static_cast<double>( maxPoint[ i ] - minPoint[ i ] );

    /** Scale the fixedImageSpacing by their ratio. */
    if ( oldLength_i > smallnumber )
    {
      imageSpacing[ i ] = this->m_ImageSpacing[ i ]
        * ( newLength_i / oldLength_i );
      finalGridSpacing[ i ] = this->m_FinalGridSpacing[ i ]
        * ( newLength_i / oldLength_i );
    }
  }

} // end ApplyInitialTransform()


/**
 * ********************* GetBSplineGrid ****************************
 */

template < typename TTransformScalarType, unsigned int VImageDimension >
void
GridScheduleComputer<TTransformScalarType, VImageDimension>
::GetBSplineGrid(
  unsigned int level,
  RegionType & gridRegion,
  SpacingType & gridSpacing,
  OriginType & gridOrigin )
{
  /** Check level. */
  if ( level > this->m_NumberOfLevels - 1 )
  {
    itkExceptionMacro(
      << "ERROR: Requesting resolution level "
      << level
      << ", but only "
      << this->m_NumberOfLevels
      << " levels exist." );
  }

  /** Return values. */
  gridRegion  = this->m_GridRegions[ level ];
  gridSpacing = this->m_GridSpacings[ level ];
  gridOrigin  = this->m_GridOrigins[ level ];

} // end GetBSplineGrid()


/**
 * ********************* PrintSelf ****************************
 */

template < typename TTransformScalarType, unsigned int VImageDimension >
void
GridScheduleComputer<TTransformScalarType, VImageDimension>
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "B-spline order: " << this->m_BSplineOrder << std::endl;
  os << indent << "NumberOfLevels: " << this->m_NumberOfLevels << std::endl;

  os << indent << "ImageSpacing: " << this->m_ImageSpacing << std::endl;
  os << indent << "ImageOrigin: " << this->m_ImageOrigin << std::endl;
  os << indent << "ImageRegion: " << std::endl;
  this->m_ImageRegion.Print( os, indent.GetNextIndent() );

  os << indent << "FinalGridSpacing: " << this->m_FinalGridSpacing << std::endl;
  os << indent << "GridSpacingFactors: " << std::endl;
  for ( unsigned int i = 0; i < this->m_NumberOfLevels; ++i )
  {
    os << indent.GetNextIndent() << this->m_GridSpacingFactors[ i ] << std::endl;
  }

  os << indent << "GridSpacings: " << std::endl;
  for ( unsigned int i = 0; i < this->m_NumberOfLevels; ++i )
  {
    os << indent.GetNextIndent() << this->m_GridSpacings[ i ] << std::endl;
  }

  os << indent << "GridOrigins: " << std::endl;
  for ( unsigned int i = 0; i < this->m_NumberOfLevels; ++i )
  {
    os << indent.GetNextIndent() << this->m_GridOrigins[ i ] << std::endl;
  }

  os << indent << "GridRegions: " << std::endl;
  for ( unsigned int i = 0; i < this->m_NumberOfLevels; ++i )
  {
    os << indent.GetNextIndent() << this->m_GridRegions[ i ] << std::endl;
  }

  os << indent << "UpsamplingFactor: " << this->m_UpsamplingFactor << std::endl;

} // end PrintSelf()


} // end namespace itk


#endif // end #ifndef __itkGridScheduleComputer_TXX__

