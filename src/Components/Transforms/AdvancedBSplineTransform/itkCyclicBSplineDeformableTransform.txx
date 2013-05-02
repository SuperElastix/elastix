/*======================================================================

  This file is part of the elastix software.

  Copyright (c) Erasmus MC University Medical Center Rotterdam.
  All rights reserved.  See src/CopyrightElastix.txt or
  http://elastix.isi.uu.nl/legal.php for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkCyclicBSplineDeformableTransform_txx
#define __itkCyclicBSplineDeformableTransform_txx

#include "itkCyclicBSplineDeformableTransform.h"
#include "itkContinuousIndex.h"
#include "itkImageRegionIterator.h"


namespace itk {

/** Constructor with default arguments. */
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
CyclicBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::CyclicBSplineDeformableTransform():Superclass()
{

}

/** Destructor. */
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
CyclicBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::~CyclicBSplineDeformableTransform()
{

}

/** Set the grid region. */
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
CyclicBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::SetGridRegion( const RegionType& region )
{
  /** Call superclass SetGridRegion. */
  Superclass::SetGridRegion( region );

  /** Check if last dimension of supportregion < last dimension of grid. */
  const int lastDim = this->m_GridRegion.GetImageDimension() - 1;
  const int lastDimSize = this->m_GridRegion.GetSize( lastDim );
  const int supportLastDimSize = this->m_SupportSize.GetElement( lastDim );
  if (supportLastDimSize > lastDimSize)
  {
    itkExceptionMacro( "Last dimension (" << lastDim << ") of support size ("
                       << supportLastDimSize << ") is larger than the "
                       << "number of grid points in the last dimension ("
                       << lastDimSize << ")." );
  }

}

/** Check if the point lies inside a valid region. */
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
bool
CyclicBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::InsideValidRegion(
  const ContinuousIndexType& index ) const
{
  bool inside = true;

  /** Check if index can be evaluated given the current grid. */
  for ( unsigned int j = 0; j < SpaceDimension - 1; j++ )
  {
    if ( index[ j ] < this->m_ValidRegionBegin[ j ] || index[ j ] >= this->m_ValidRegionEnd[ j ] ) {
      inside = false;
      break;
    }
  }

  return inside;
}

/** Split region into two parts: 1) The part that reaches from
 * inRegion.index to the border of the inImage in the last dimension and
 * 2) The part that reaches from 0 in the last dimension to the end of the
 * inRegion.
 */
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
CyclicBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::SplitRegion(
  const RegionType & imageRegion,
  const RegionType & inRegion,
  RegionType & outRegion1,
  RegionType & outRegion2) const
{
  /** Set initial index and sizes of the two regions. */
  IndexType index1 = inRegion.GetIndex();
  IndexType index2 = inRegion.GetIndex();

  SizeType size1 = inRegion.GetSize();
  SizeType size2;
  size2.Fill( 0 );

  /** Get last dimension information. */
  const unsigned int lastDim = imageRegion.GetImageDimension() - 1;
  const unsigned int lastDimSize = imageRegion.GetSize( lastDim );
  const unsigned int supportLastDimSize = inRegion.GetSize( lastDim );

  /** Check if we need to split. */
  const int lastDimIndex = inRegion.GetIndex( lastDim );
  if ( lastDimIndex < 0 )
  {
    /** Set new index and size for supportRegion1. */
    index1.SetElement( lastDim, lastDimSize + lastDimIndex );
    size1.SetElement( lastDim, abs( lastDimIndex ) );

    /** Set new index and size for supportRegion2. */
    index2.SetElement( lastDim, 0 );
    size2 = inRegion.GetSize();
    size2.SetElement( lastDim, supportLastDimSize + lastDimIndex );
  }
  else if ( lastDimIndex + supportLastDimSize > lastDimSize )
  {
    /** Set last dimension item of index2 to zero. */
    index2.SetElement( lastDim, 0 );

    /** Set new size of supportRegion1. */
    size1.SetElement( lastDim, lastDimSize - lastDimIndex );

    /** Set size and index of supportRegion2. */
    size2 = inRegion.GetSize();
    size2.SetElement( lastDim, supportLastDimSize - size1.GetElement( lastDim ) );
  }

  /** Set region indices and sizes. */
  outRegion1.SetIndex(index1);
  outRegion1.SetSize(size1);
  outRegion2.SetIndex(index2);
  outRegion2.SetSize(size2);
}

/** Transform a point. */
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
CyclicBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::TransformPoint(
  const InputPointType & point,
  OutputPointType & outputPoint,
  WeightsType & weights,
  ParameterIndexArrayType & indices,
  bool & inside ) const
{
  inside = true;
  InputPointType transformedPoint = point;

  /** Check if the coefficient image has been set. */
  if ( !this->m_CoefficientImages[ 0 ] )
  {
    itkWarningMacro( << "B-spline coefficients have not been set" );
    for ( unsigned int j = 0; j < SpaceDimension; j++ )
    {
      outputPoint[ j ] = transformedPoint[ j ];
    }
    return;
  }

  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( point, cindex );

  /** NOTE: if the support region does not lie totally within the grid
   * (except for the last dimension, which wraps around) we assume
   * zero displacement and return the input point.
   */
  inside = this->InsideValidRegion( cindex );
  if ( !inside )
  {
    outputPoint = transformedPoint;
    return;
  }

  /** Compute interpolation weights. */
  IndexType supportIndex;
  this->m_WeightsFunction->ComputeStartIndex( cindex, supportIndex );
  this->m_WeightsFunction->Evaluate( cindex, supportIndex, weights );

  /** For each dimension, correlate coefficient with weights. */
  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  supportRegion.SetIndex( supportIndex );

  /** Split support region into two parts. */
  RegionType supportRegions[ 2 ];
  this->SplitRegion( this->m_CoefficientImages[ 0 ]->GetLargestPossibleRegion(),
    supportRegion, supportRegions[ 0 ], supportRegions[ 1 ] );

  /** Zero output point elements. */
  outputPoint.Fill( NumericTraits<ScalarType>::Zero );

  unsigned long counter = 0;
  for ( unsigned int r = 0; r < 2; ++r)
  {
    /** Create iterators over the coefficient images
     * (for both supportRegion1 and supportRegion2.
     */
    typedef ImageRegionConstIterator<ImageType> IteratorType;
    IteratorType iterator[ SpaceDimension ];

    const PixelType * basePointer
      = this->m_CoefficientImages[ 0 ]->GetBufferPointer();

    for ( unsigned int j = 0; j < SpaceDimension - 1; j++ )
    {
      iterator[ j ] = IteratorType( this->m_CoefficientImages[ j ], supportRegions[ r ] );
    }

    /** Loop over this support region. */
    while ( !iterator[ 0 ].IsAtEnd() )
    {
      /** Populate the indices array. */
      indices[ counter ] = &(iterator[ 0 ].Value()) - basePointer;

      /** Multiply weigth with coefficient to compute displacement. */
      for ( unsigned int j = 0; j < SpaceDimension - 1; j++ )
      {
         outputPoint[ j ] += static_cast<ScalarType>(
           weights[ counter ] * iterator[ j ].Value() );
         ++iterator[ j ];
      }
      ++ counter;

    } // end while
  }

  /** The output point is the start point + displacement. */
  for ( unsigned int j = 0; j < SpaceDimension; j++ )
  {
    outputPoint[ j ] += transformedPoint[ j ];
  }
}


/** Compute the Jacobian in one position. */
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
CyclicBSplineDeformableTransform<TScalarType, NDimensions,VSplineOrder>
::GetJacobian( const InputPointType & point, WeightsType& weights, ParameterIndexArrayType& indexes) const
{
  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  const PixelType * basePointer = this->m_CoefficientImages[0]->GetBufferPointer();

  /** Tranform from world coordinates to grid coordinates. */
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( point, cindex );

  /** NOTE: if the support region does not lie totally within the grid
       * we assume zero displacement and return the input point.
       */
  if ( !this->InsideValidRegion( cindex ) )
  {
    weights.Fill(0.0);
    indexes.Fill(0);
    return;
  }

  /** Compute interpolation weights. */
  IndexType supportIndex;
  this->m_WeightsFunction->ComputeStartIndex( cindex, supportIndex );
  this->m_WeightsFunction->Evaluate( cindex, supportIndex, weights );

  supportRegion.SetIndex( supportIndex );
  /** Split support region into two parts. */
  RegionType supportRegions[ 2 ];
  this->SplitRegion( this->m_CoefficientImages[ 0 ]->GetLargestPossibleRegion(),
    supportRegion, supportRegions[ 0 ], supportRegions[ 1 ] );

  /** For each dimension, copy the weight to the support region. */
  unsigned long counter = 0;
  typedef ImageRegionIterator<JacobianImageType> IteratorType;
  for ( unsigned int r = 0; r < 2; ++r )
  {
    IteratorType iterator = IteratorType( this->m_CoefficientImages[ 0 ], supportRegions[ r ] );

    while ( ! iterator.IsAtEnd() )
    {
      indexes[ counter ] = &( iterator.Value() ) - basePointer;

      /** Go to next coefficient in the support region. */
      ++ counter;
      ++ iterator;
    }
  }
}


/**
 * ********************* GetSpatialJacobian ****************************
 */

template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
CyclicBSplineDeformableTransform<TScalarType, NDimensions,VSplineOrder>
::GetSpatialJacobian(
  const InputPointType & ipp,
  SpatialJacobianType & sj ) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if ( this->m_InputParametersPointer == NULL )
  {
    itkExceptionMacro( << "Cannot compute Jacobian: parameters not set" );
  }

  /** Convert the physical point to a continuous index, which
       * is needed for the 'Evaluate()' functions below.
       */
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( ipp, cindex );

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and identity spatial Jacobian
  if ( !this->InsideValidRegion( cindex ) )
  {
    sj.SetIdentity();
    return;
  }

  /** Compute the number of affected B-spline parameters. */
  /** Allocate memory on the stack: */
  const SizeValueType numberOfWeights = WeightsFunctionType::NumberOfWeights;
  typename WeightsType::ValueType weightsArray[ numberOfWeights ];
  WeightsType weights( weightsArray, numberOfWeights, false );

  IndexType supportIndex;
  this->m_DerivativeWeightsFunctions[ 0 ]->ComputeStartIndex(
    cindex, supportIndex );
  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  supportRegion.SetIndex( supportIndex );

  /** Split support region into two parts. */
  RegionType supportRegions[ 2 ];
  this->SplitRegion( this->m_CoefficientImages[ 0 ]->GetLargestPossibleRegion(),
     supportRegion, supportRegions[ 0 ], supportRegions[ 1 ] );

  sj.Fill( 0.0 );

  /** Compute the spatial Jacobian sj:
       *    dT_{dim} / dx_i = delta_{dim,i} + \sum coefs_{dim} * weights.
       */
  for ( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    /** Compute the derivative weights. */
    this->m_DerivativeWeightsFunctions[ i ]->Evaluate( cindex, supportIndex, weights );

    /** Compute the spatial Jacobian sj:
            *    dT_{dim} / dx_i = \sum coefs_{dim} * weights.
            */
    for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
    {
      /** Compute the sum for this dimension. */
      double sum = 0.0;

      typename WeightsType::const_iterator itWeights = weights.begin();

      for ( unsigned int r = 0; r < 2; ++r)
      {
        /** Create an iterator over the correct part of the coefficient
                       * image. Create an iterator over the weights vector.
                       */
        ImageRegionConstIterator<ImageType> itCoef(
          this->m_CoefficientImages[ dim ], supportRegions[ r ] );

        while ( !itCoef.IsAtEnd() )
        {
          sum += itCoef.Value() * (*itWeights);
          ++itWeights;
          ++itCoef;
        }
      } // end for r

      /** Update the spatial Jacobian sj. */
      sj( dim, i ) += sum;
    } // end for dim
  } // end for i

  /** Take into account grid spacing and direction cosines. */
  sj = sj * this->m_PointToIndexMatrix;

  /** Add identity. */
  for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
  {
    sj( dim, dim ) += 1.0;
  }

} // end GetSpatialJacobian()


/**
 * ********************* ComputeNonZeroJacobianIndices ****************************
 */

template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
CyclicBSplineDeformableTransform<TScalarType, NDimensions,VSplineOrder>
::ComputeNonZeroJacobianIndices(
  NonZeroJacobianIndicesType & nonZeroJacobianIndices,
  const RegionType & supportRegion ) const
{
  nonZeroJacobianIndices.resize( this->GetNumberOfNonZeroJacobianIndices() );

  /** Split support region into two parts. */
  RegionType supportRegions[ 2 ];
  this->SplitRegion( this->m_CoefficientImages[ 0 ]->GetLargestPossibleRegion(),
    supportRegion, supportRegions[ 0 ], supportRegions[ 1 ] );

  /** Initialize some helper variables. */
  const SizeValueType numberOfWeights = WeightsFunctionType::NumberOfWeights;
  const SizeValueType parametersPerDim
    = this->GetNumberOfParametersPerDimension();
  unsigned long mu = 0;

  for ( unsigned int r = 0; r < 2; ++r ) {
    /** Create iterator over the coefficient image (for current supportRegion). */
    ImageRegionConstIteratorWithIndex< ImageType >
      iterator( this->m_CoefficientImages[ 0 ], supportRegions[ r ] );

    /** For all control points in the support region, set which of the
            * indices in the parameter array are non-zero.
            */
    const PixelType * basePointer = this->m_CoefficientImages[ 0 ]->GetBufferPointer();
    while ( !iterator.IsAtEnd() )
    {
      /** Translate the index into a parameter number for the x-direction. */
      const IdentifierType parameterNumber = &(iterator.Value()) - basePointer;

      /** Update the nonZeroJacobianIndices for all directions. */
      for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
      {
        nonZeroJacobianIndices[ mu + dim * numberOfWeights ]
          = parameterNumber + dim * parametersPerDim;
      }

      /** Increase the iterators. */
      ++iterator;
      ++mu;
    } // end while
  } // end for (supportregions)

} // end ComputeNonZeroJacobianIndices()

} // namespace

#endif
