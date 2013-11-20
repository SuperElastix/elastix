/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkCyclicGridScheduleComputer_hxx__
#define __itkCyclicGridScheduleComputer_hxx__

#include "itkCyclicGridScheduleComputer.h"
#include "itkConfigure.h"

#include "itkImageRegionExclusionConstIteratorWithIndex.h"


namespace itk
{

/**
 * ********************* Constructor ****************************
 */

template < typename TTransformScalarType, unsigned int VImageDimension >
CyclicGridScheduleComputer<TTransformScalarType, VImageDimension>
::CyclicGridScheduleComputer()
{

} // end Constructor()


/**
 * ********************* ComputeBSplineGrid ****************************
 */

template < typename TTransformScalarType, unsigned int VImageDimension >
void
CyclicGridScheduleComputer<TTransformScalarType, VImageDimension>
::ComputeBSplineGrid( void )
{
  /** Call superclass method. */
  //Superclass::ComputeBSplineGrid();

  OriginType imageOrigin;
  SpacingType imageSpacing, finalGridSpacing;
  DirectionType imageDirection;

  /** Apply the initial transform. */
  this->ApplyInitialTransform( imageOrigin, imageSpacing, imageDirection, finalGridSpacing );

  /** Set the appropriate sizes. */
  this->m_GridOrigins.resize( this->GetNumberOfLevels() );
  this->m_GridRegions.resize( this->GetNumberOfLevels() );
  this->m_GridSpacings.resize( this->GetNumberOfLevels() );
  this->m_GridDirections.resize( this->GetNumberOfLevels() );

  /** For all levels ... */
  for ( unsigned int res = 0; res < this->GetNumberOfLevels(); ++res )
  {
    /** For all dimensions ... */
    SizeType size = this->GetImageRegion().GetSize();
    SizeType gridsize;
    for ( unsigned int dim = 0; dim < Dimension; ++dim )
    {
      /** Compute the grid spacings. */
      double gridSpacing
        = finalGridSpacing[ dim ] * this->m_GridSpacingFactors[ res ][ dim ];

      /** Check if the grid spacing matches the Cyclic behaviour of this
       * transform. We want the spacing at the borders for the last dimension
       * to be half the grid spacing.
       */
      unsigned int bareGridSize = 0;
      if (dim == Dimension - 1)
      {
        const float lastDimSizeInPhysicalUnits =
               imageSpacing[ dim ] * this->GetImageRegion().GetSize( dim );

        /** Compute closest correct spacing. */

        /** Compute number of nodes. */
        bareGridSize = static_cast<unsigned int>( lastDimSizeInPhysicalUnits /
                                                                   gridSpacing );

        /** Compute new (larger) gridspacing. */
        gridSpacing =
              lastDimSizeInPhysicalUnits / static_cast<float> ( bareGridSize );

      }
      else
      {
        /** Compute the grid size without the extra grid points at the edges. */
        bareGridSize = static_cast<unsigned int>(
          vcl_ceil( size[ dim ] * imageSpacing[ dim ] / gridSpacing ) );
      }

      this->m_GridSpacings[ res ][ dim ] = gridSpacing;

      /** The number of B-spline grid nodes is the bareGridSize plus the
       * B-spline order more grid nodes (for all dimensions but the last).
       * The last dimension has the bareGridSize.
       */
      gridsize[ dim ] = static_cast<SizeValueType>( bareGridSize );
      if (dim < Dimension - 1)
      {
        gridsize[ dim ] += static_cast<SizeValueType> ( this->GetBSplineOrder() );
      }

      /** Compute the origin of the B-spline grid. */
      this->m_GridOrigins[ res ][ dim ] = imageOrigin[ dim ] -
        ( ( gridsize[ dim ] - 1 ) * gridSpacing
        - ( size[ dim ] - 1 ) * imageSpacing[ dim ] ) / 2.0;
    }

    /** Take into account direction cosines:
     * rotate grid origin around image origin. */
    this->m_GridOrigins[ res ] = imageOrigin + imageDirection * (
      this->m_GridOrigins[ res ] - imageOrigin );

    /** Set the grid region. */
    this->m_GridRegions[ res ].SetSize( gridsize );

    /** Simply copy the image direction for now */
    this->m_GridDirections[ res ] = imageDirection;
  }

} // end ComputeBSplineGrid()

} // end namespace itk

#endif // end #ifndef __itkCyclicGridScheduleComputer_hxx__
