/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkTransformBendingEnergyPenaltyTerm_txx
#define __itkTransformBendingEnergyPenaltyTerm_txx

#include "itkTransformBendingEnergyPenaltyTerm.h"


namespace itk
{

/**
 * ****************** Constructor *******************************
 */

template< class TFixedImage, class TScalarType >
TransformBendingEnergyPenaltyTerm< TFixedImage, TScalarType >
::TransformBendingEnergyPenaltyTerm()
{
  /** Initialize member variables. */
} // end constructor


/**
 * ****************** PrintSelf *******************************
 *

template< class TFixedImage, class TScalarType >
void
TransformBendingEnergyPenaltyTerm< TFixedImage, TScalarType >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  this->Superclass::PrintSelf( os, indent );

  //     os << indent << "Transform: "
  //       << this->m_Transform->GetPointer() << std::endl;

} // end PrintSelf()


/**
 * ****************** GetValue *******************************
 */

template< class TFixedImage, class TScalarType >
typename TransformBendingEnergyPenaltyTerm< TFixedImage, TScalarType >::MeasureType
TransformBendingEnergyPenaltyTerm< TFixedImage, TScalarType >
::GetValue( const ParametersType & parameters ) const
{
  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  RealType measure = NumericTraits<RealType>::Zero;
  SpatialHessianType spatialHessian;

  /** Check if the SpatialHessian is nonzero.
   * If it is zero, it was implemented in the transform as such, which
   * by convention means that it is zero everywhere. Therefore, this
   * penalty term evaluates to zero, so we return zero before looping over
   * all samples.
   * We check by computing the spatial Hessian for the center coordinate.
   */
  FixedImagePointType tmpPoint;
  typename FixedImageType::IndexType index;
  typename FixedImageType::SizeType size
    = this->GetFixedImage()->GetLargestPossibleRegion().GetSize();
  for ( unsigned int i = 0; i < FixedImageDimension; ++i )
  {
    index[ i ] = size[ i ] / 2;
  }
  this->GetFixedImage()->TransformIndexToPhysicalPoint( index, tmpPoint );
  this->m_AdvancedTransform->GetSpatialHessian( tmpPoint, spatialHessian );
  if ( spatialHessian.Size() == 0 )
  {
    return static_cast<MeasureType>( measure );
  }

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  /** Update the imageSampler and get a handle to the sample container. */
  this->GetImageSampler()->Update();
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image samples to calculate the penalty term. */
  for ( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
    MovingImagePointType mappedPoint;

    /** Transform point and check if it is inside the bspline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if ( sampleOk ) 
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );        
    }

    if ( sampleOk )
    {
      this->m_NumberOfPixelsCounted++; 

      /** Get the spatial Hessian of the transformation at the current point.
       * This is needed to compute the bending energy.
       */
      this->m_AdvancedTransform->GetSpatialHessian( fixedPoint, spatialHessian );

      /** Compute the contribution of this point. */
      for ( unsigned int k = 0; k < FixedImageDimension; ++k )
      {
        measure += vnl_math_sqr(
          spatialHessian[ k ].GetVnlMatrix().frobenius_norm() );
      }

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Update measure value. */
  measure /= static_cast<RealType>( this->m_NumberOfPixelsCounted );

  /** Return the value. */
  return static_cast<MeasureType>( measure );

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template< class TFixedImage, class TScalarType >
void
TransformBendingEnergyPenaltyTerm< TFixedImage, TScalarType >
::GetDerivative(
  const ParametersType & parameters,
  DerivativeType & derivative ) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable.
   */
  MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
  this->GetValueAndDerivative( parameters, dummyvalue, derivative );
  
} // end GetDerivative()


/**
 * ****************** GetValueAndDerivative *******************************
 */

template< class TFixedImage, class TScalarType >
void
TransformBendingEnergyPenaltyTerm< TFixedImage, TScalarType >
::GetValueAndDerivative(
  const ParametersType & parameters,
  MeasureType & value,
  DerivativeType & derivative ) const
{
  /** Create and initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  RealType measure = NumericTraits< RealType >::Zero;
  derivative = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< DerivativeValueType >::Zero );

  SpatialHessianType spatialHessian;
  JacobianOfSpatialHessianType jacobianOfSpatialHessian;
  NonZeroJacobianIndicesType nonZeroJacobianIndices;

  /** Check if the SpatialHessian and the JacobianOfSpatialHessian are nonzero.
   * If they are zero, they were implemented in the transform as such, which
   * by convention means that they are zero everywhere. Therefore, this
   * penalty term evaluates to zero, so we return zero before looping over
   * all samples.
   * We check by computing the spatial Hessian for the center coordinate.
   */
  FixedImagePointType tmpPoint;
  typename FixedImageType::IndexType index;
  typename FixedImageType::SizeType size
    = this->GetFixedImage()->GetLargestPossibleRegion().GetSize();
  for ( unsigned int i = 0; i < FixedImageDimension; ++i )
  {
    index[ i ] = size[ i ] / 2;
  }
  this->GetFixedImage()->TransformIndexToPhysicalPoint( index, tmpPoint );
  this->m_AdvancedTransform->GetSpatialHessian( tmpPoint, spatialHessian );
  this->m_AdvancedTransform->GetJacobianOfSpatialHessian( tmpPoint,
    jacobianOfSpatialHessian, nonZeroJacobianIndices );
  if ( spatialHessian.Size() == 0 && jacobianOfSpatialHessian.size() == 0 )
  {
    value = static_cast<MeasureType>( measure );
    return;
  }
  // TODO: This is only required once! and not every iteration.

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  /** Check if this transform is a B-spline transform. */
  bool transformIsBSpline = this->m_TransformIsAdvancedBSpline
    || this->m_TransformIsAdvancedBSplineCombination;
  // \todo: Check if Combo transform is also ok for speedup trick.

  /** Update the imageSampler and get a handle to the sample container. */
  this->GetImageSampler()->Update();
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image to calculate the penalty term and its derivative. */
  for ( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
    MovingImagePointType mappedPoint;

    /** Although the mapped point is not needed to compute the penalty term,
     * we compute in order to check if it maps inside the support region of
     * the B-spline and if it maps inside the moving image mask.
     */

    /** Transform point and check if it is inside the bspline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if ( sampleOk ) 
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );        
    }

    // \todo Need to check if it maps in the moving image ??:
    // well it maps in the moving mask anyway.
//     MovingImageContinuousIndexType cindex;
//     if ( sampleOk )
//     {//       this->m_Interpolator->ConvertPointToContinuousIndex( mappedPoint, cindex );//       sampleOk = this->m_Interpolator->IsInsideBuffer( cindex );//     }
    if ( sampleOk )
    {
      this->m_NumberOfPixelsCounted++; 

      /** Get the spatial Hessian of the transformation at the current point.
       * This is needed to compute the bending energy.
       */
      this->m_AdvancedTransform->GetJacobianOfSpatialHessian( fixedPoint,
        spatialHessian, jacobianOfSpatialHessian, nonZeroJacobianIndices );

      /** Compute the contribution to the metric value of this point. */
      for ( unsigned int k = 0; k < FixedImageDimension; ++k )
      {
        measure += vnl_math_sqr(
          spatialHessian[ k ].GetVnlMatrix().frobenius_norm() );
      }

      /** Prepare some stuff for the computation of the metric derivative. */
      std::vector< const InternalMatrixType > A( FixedImageDimension );
      for ( unsigned int k = 0; k < FixedImageDimension; ++k )
      {
        A[ k ] = spatialHessian[ k ].GetVnlMatrix();
      }
      const RealType Bsize = static_cast<RealType>(
        jacobianOfSpatialHessian[ 0 ][ 0 ].GetVnlMatrix().size() );

      /** Make a distinction between a B-spline transform and other transforms. */
      if ( !transformIsBSpline )
      {
        /** Compute the contribution to the metric derivative of this point. */
        for ( unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu )
        {
          for ( unsigned int k = 0; k < FixedImageDimension; ++k )
          {
            /** This computes:
             * \sum_i \sum_j A_ij B_ij = element_product(A,B).mean()*B.size()
             */
            const InternalMatrixType & B
              = jacobianOfSpatialHessian[ mu ][ k ].GetVnlMatrix();
            const RealType matrixMean = element_product( A[ k ], B ).mean();
            derivative[ nonZeroJacobianIndices[ mu ] ]
              += 2.0 * matrixMean * Bsize;
          }
        }
      }
      else
      {
        /** For the B-spline transform we know that only 1/FixedImageDimension
         * part of the JacobianOfSpatialHessian is non-zero.
         */

        /** Compute the contribution to the metric derivative of this point. */
        unsigned int numParPerDim
          = nonZeroJacobianIndices.size() / FixedImageDimension;
        for ( unsigned int mu = 0; mu < numParPerDim; ++mu )
        {
          for ( unsigned int k = 0; k < FixedImageDimension; ++k )
          {
            /** This computes:
             * \sum_i \sum_j A_ij B_ij = element_product(A,B).mean()*B.size()
             */
            const InternalMatrixType & B
              = jacobianOfSpatialHessian[ mu + numParPerDim * k ][ k ].GetVnlMatrix();
            const RealType matrixMean = element_product( A[ k ], B ).mean();
            derivative[ nonZeroJacobianIndices[ mu + numParPerDim * k ] ]
              += 2.0 * matrixMean * Bsize;
          }
        }
      } // end if B-spline

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Update measure value. */
  measure /= static_cast<RealType>( this->m_NumberOfPixelsCounted );
  derivative /= static_cast<RealType>( this->m_NumberOfPixelsCounted );

  /** The return value. */
  value = static_cast<MeasureType>( measure );

} // end GetValueAndDerivative()


} // end namespace itk

#endif // #ifndef __itkTransformBendingEnergyPenaltyTerm_txx

