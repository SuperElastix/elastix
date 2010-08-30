/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkAdvancedTransformRigidityPenaltyTerm_txx
#define __itkAdvancedTransformRigidityPenaltyTerm_txx

#include "itkAdvancedTransformRigidityPenaltyTerm.h"

#include "vnl/vnl_fastops.h"
#include "vnl/vnl_det.h"
#include "vnl/vnl_trace.h"
#include "vnl/algo/vnl_adjugate.h"


namespace itk
{

/**
 * ****************** Constructor *******************************
 */

template< class TFixedImage, class TScalarType >
AdvancedTransformRigidityPenaltyTerm< TFixedImage, TScalarType >
::AdvancedTransformRigidityPenaltyTerm()
{
  /** Weights. */
  this->m_LinearityConditionWeight      = NumericTraits<ScalarType>::One;
  this->m_OrthonormalityConditionWeight = NumericTraits<ScalarType>::One;
  this->m_PropernessConditionWeight     = NumericTraits<ScalarType>::One;

  /** Usage. */
  this->m_UseLinearityCondition             = true;
  this->m_UseOrthonormalityCondition        = true;
  this->m_UsePropernessCondition            = true;
  this->m_CalculateLinearityCondition       = true;
  this->m_CalculateOrthonormalityCondition  = true;
  this->m_CalculatePropernessCondition      = true;

  /** Values. */
  this->m_LinearityConditionValue       = NumericTraits<MeasureType>::Zero;
  this->m_OrthonormalityConditionValue  = NumericTraits<MeasureType>::Zero;
  this->m_PropernessConditionValue      = NumericTraits<MeasureType>::Zero;

  /** Turn on the sampler functionality. */
  this->SetUseImageSampler( true );

} // end constructor


/**
 * ****************** GetValue *******************************
 */

template< class TFixedImage, class TScalarType >
typename AdvancedTransformRigidityPenaltyTerm< TFixedImage, TScalarType >::MeasureType
AdvancedTransformRigidityPenaltyTerm< TFixedImage, TScalarType >
::GetValue( const ParametersType & parameters ) const
{
  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits<MeasureType>::Zero;
  SpatialJacobianType spatialJacobian;
  SpatialHessianType spatialHessian;
  this->m_LinearityConditionValue       = NumericTraits<MeasureType>::Zero;
  this->m_OrthonormalityConditionValue  = NumericTraits<MeasureType>::Zero;
  this->m_PropernessConditionValue      = NumericTraits<MeasureType>::Zero;

  /** Check if the SpatialHessian is nonzero. */
//   if ( !this->m_AdvancedTransform->GetHasNonZeroSpatialHessian() )
//   {
//     return measure;
//   }

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

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if ( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    if ( sampleOk )
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the spatial Jacobian and Hessian of the transformation at the
       * current point. The spatial Jacobian is needed to compute OC and PC.
       * The spatial Hessian is needed to compute LC.
       */
      this->m_AdvancedTransform->GetSpatialJacobian( fixedPoint, spatialJacobian );
      this->m_AdvancedTransform->GetSpatialHessian( fixedPoint, spatialHessian );

      /** Compute the contribution of this point to LC. */
      if ( this->m_CalculateLinearityCondition )
      {
        for ( unsigned int k = 0; k < FixedImageDimension; ++k )
        {
          this->m_LinearityConditionValue += vnl_math_sqr(
            spatialHessian[ k ].GetVnlMatrix().frobenius_norm() );
        }
      }

      /** Compute the contribution of this point to OC. */
      if ( this->m_CalculateOrthonormalityCondition )
      {
        typename SpatialJacobianType::InternalMatrixType tmp;
        //vnl_fastops::AtA( tmp, spatialJacobian.GetVnlMatrix() );
        this->m_OrthonormalityConditionValue += vnl_math_sqr( tmp.frobenius_norm() );
      }

      /** Compute the contribution of this point to PC. */
      if ( this->m_CalculatePropernessCondition )
      {
        MeasureType tmp = vnl_det( spatialJacobian.GetVnlMatrix() ) - 1.0;
        this->m_PropernessConditionValue += vnl_math_sqr( tmp );
      }

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Add all parts together to form the total rigidity term. */
  if ( this->m_UseLinearityCondition )
  {
    measure += this->m_LinearityConditionWeight
      * this->m_LinearityConditionValue;
  }
  if ( this->m_UseOrthonormalityCondition )
  {
    measure += this->m_OrthonormalityConditionWeight
      * this->m_OrthonormalityConditionValue;
  }
  if ( this->m_UsePropernessCondition )
  {
    measure += this->m_PropernessConditionWeight
      * this->m_PropernessConditionValue;
  }
  measure /= static_cast<MeasureType>( this->m_NumberOfPixelsCounted );

  /** Return the value. */
  return measure;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template< class TFixedImage, class TScalarType >
void
AdvancedTransformRigidityPenaltyTerm< TFixedImage, TScalarType >
::GetDerivative(
  const ParametersType & parameters,
  DerivativeType & derivative ) const
{
  /** Slower, but works. */
  MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
  this->GetValueAndDerivative( parameters, dummyvalue, derivative );

} // end GetDerivative()


/**
 * ****************** GetValueAndDerivative *******************************
 */

template< class TFixedImage, class TScalarType >
void
AdvancedTransformRigidityPenaltyTerm< TFixedImage, TScalarType >
::GetValueAndDerivative(
  const ParametersType & parameters,
  MeasureType & value,
  DerivativeType & derivative ) const
{
  itkExceptionMacro( << "ERROR: Don't use the GetValueAndDerivative() of this class. It's not functional yet. " );

  /** Create and initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  RealType measure = NumericTraits< RealType >::Zero;
  derivative = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< DerivativeValueType >::Zero );

  SpatialJacobianType spatialJacobian;
  SpatialHessianType spatialHessian;
  JacobianOfSpatialHessianType jacobianOfSpatialHessian;
  NonZeroJacobianIndicesType nonZeroJacobianIndices;
  unsigned long numberOfNonZeroJacobianIndices = this->m_AdvancedTransform
     ->GetNumberOfNonZeroJacobianIndices();
  jacobianOfSpatialHessian.resize( numberOfNonZeroJacobianIndices );
  nonZeroJacobianIndices.resize( numberOfNonZeroJacobianIndices );

  /** Check if the SpatialHessian is nonzero. */
  if ( !this->m_AdvancedTransform->GetHasNonZeroSpatialHessian()
    && !this->m_AdvancedTransform->GetHasNonZeroJacobianOfSpatialHessian() )
  {
    value = static_cast<MeasureType>( measure );
    return;
  }
  // TODO: This is only required once! and not every iteration.

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  /** Check if this transform is a B-spline transform. */
//   typename BSplineTransformType::Pointer dummy = 0;
//   bool transformIsBSpline = this->CheckForBSplineTransform( dummy );

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

    /** Transform point and check if it is inside the B-spline support region. */
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
//       this->m_AdvancedTransform->GetSpatialHessian( fixedPoint,
//         spatialHessian );
//       this->m_AdvancedTransform->GetJacobianOfSpatialHessian( fixedPoint,
//         jacobianOfSpatialHessian, nonZeroJacobianIndices );
      this->m_AdvancedTransform->GetSpatialJacobian( fixedPoint, spatialJacobian );
       this->m_AdvancedTransform->GetJacobianOfSpatialHessian( fixedPoint,
         spatialHessian, jacobianOfSpatialHessian, nonZeroJacobianIndices );

//       /** Prepare some stuff for the computation of the metric (derivative). */
//       FixedArray< InternalMatrixType, FixedImageDimension > A;
//       for ( unsigned int k = 0; k < FixedImageDimension; ++k )
//       {
//         A[ k ] = spatialHessian[ k ].GetVnlMatrix();
//       }

      /** Compute the contribution of this point to LC. */
      if ( this->m_CalculateLinearityCondition )
      {
        for ( unsigned int k = 0; k < FixedImageDimension; ++k )
        {
          this->m_LinearityConditionValue += vnl_math_sqr(
            spatialHessian[ k ].GetVnlMatrix().frobenius_norm() );
        }

        /** Compute the contribution to the metric derivative of this point. */
        for ( unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu )
        {
          const SpatialHessianType & sh = jacobianOfSpatialHessian[ mu ];

          for ( unsigned int k = 0; k < FixedImageDimension; ++k )
          {
            /** \sum_i \sum_j A_ij B_ij = element_product(A,B).mean()*B.size() */
            const InternalMatrixType & B = sh[ k ].GetVnlMatrix();

            RealType matrixProduct
              = element_product( spatialHessian[ k ].GetVnlMatrix(), B ).mean() * B.size();
            derivative[ nonZeroJacobianIndices[ mu ] ] += 2.0 * matrixProduct;
          }
        }
      } // end if linearity

      /** Compute the contribution of this point to OC. */
      if ( this->m_CalculateOrthonormalityCondition )
      {
        typename SpatialJacobianType::InternalMatrixType tmp;
        //vnl_fastops::AtA( tmp, spatialJacobian.GetVnlMatrix() );
        this->m_OrthonormalityConditionValue += vnl_math_sqr( tmp.frobenius_norm() );

        /** Compute the contribution of this point to OC derivative . */
        for ( unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu )
        {
          for ( unsigned int k = 0; k < FixedImageDimension; ++k )
          {
            const InternalMatrixType & B
              = jacobianOfSpatialHessian[ mu ][ k ].GetVnlMatrix();
            //vnl_fastops::AtB( tmp, spatialJacobian.GetVnlMatrix(), B );
            derivative[ nonZeroJacobianIndices[ mu ] ] += 2.0 * tmp.mean() * B.size();
          }
        }
      } // end if orthonormality

      /** Compute the contribution of this point to PC. */
      if ( this->m_CalculatePropernessCondition )
      {
        MeasureType tmp = vnl_det( spatialJacobian.GetVnlMatrix() ) - 1.0;
        this->m_PropernessConditionValue += vnl_math_sqr( tmp );

        /** Compute the contribution of this point to OC derivative . */
        for ( unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu )
        {
          typename SpatialJacobianType::InternalMatrixType tmp1;
          typename SpatialJacobianType::InternalMatrixType tmp2;
          //tmp1 = vnl_adjugate( spatialJacobian.GetVnlMatrix() );
          //vnl_fastops::AB( tmp2, tmp, jacobianOfSpatialJacobian[ mu ].GetVnlMatrix() );
          derivative[ nonZeroJacobianIndices[ mu ] ] += 2.0 *
            ( vnl_det( spatialJacobian.GetVnlMatrix() ) - 1.0 ) * vnl_trace( tmp2 );
        }
      }

//       /** Compute the contribution to the metric value of this point. */
//       for ( unsigned int k = 0; k < FixedImageDimension; ++k )
//       {
//         measure += vnl_math_sqr( A[ k ].frobenius_norm() );
//       }
//
//       /** Make a distinction between a B-spline transform and other transforms. */
//       if ( !transformIsBSpline )
//       {
//         /** Compute the contribution to the metric derivative of this point. */
//         for ( unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu )
//         {
//           const SpatialHessianType & sh = jacobianOfSpatialHessian[ mu ];
//
//           for ( unsigned int k = 0; k < FixedImageDimension; ++k )
//           {
//             /** This computes LC:
//              * \sum_i \sum_j A_ij B_ij = element_product(A,B).mean()*B.size()
//              */
//             const InternalMatrixType & B = sh[ k ].GetVnlMatrix();
//
//             RealType matrixProduct
//               = element_product( spatialHessian[ k ], B ).mean() * B.size()
//
//             RealType matrixProduct = 0.0;
//             typename InternalMatrixType::const_iterator itA = A[ k ].begin();
//             typename InternalMatrixType::const_iterator itB = B.begin();
//             typename InternalMatrixType::const_iterator itAend = A[ k ].end();
//             while ( itA != itAend )
//             {
//               matrixProduct += (*itA) * (*itB);
//               ++itA;
//               ++itB;
//             }
//
//             derivative[ nonZeroJacobianIndices[ mu ] ] += 2.0 * matrixProduct;
//           }
//         }
//       }
//       else
//       {
//         /** For the B-spline transform we know that only 1/FixedImageDimension
//          * part of the JacobianOfSpatialHessian is non-zero.
//          */
//
//         /** Compute the contribution to the metric derivative of this point. */
//         unsigned int numParPerDim
//           = nonZeroJacobianIndices.size() / FixedImageDimension;
//         SpatialHessianType * basepointer1 = &jacobianOfSpatialHessian[ 0 ];
//         unsigned long * basepointer2 = &nonZeroJacobianIndices[ 0 ];
//         double * basepointer3 = &derivative[ 0 ];*
//         for ( unsigned int mu = 0; mu < numParPerDim; ++mu )
//         {
//           for ( unsigned int k = 0; k < FixedImageDimension; ++k )
//           {
//             /** This computes:
//              * \sum_i \sum_j A_ij B_ij = element_product(A,B).mean()*B.size()
//              */
//             const InternalMatrixType & B
//               = (*( basepointer1 + mu + numParPerDim * k ))[ k ].GetVnlMatrix();
//             const RealType matrixMean = element_product( A[ k ], B ).mean();
//             *( basepointer3 + (*( basepointer2 + mu + numParPerDim * k )) )
//               += 2.0 * matrixMean * Bsize;*
//             const InternalMatrixType & B
//               = jacobianOfSpatialHessian[ mu + numParPerDim * k ][ k ].GetVnlMatrix();
//
//             RealType matrixElementProduct = 0.0;
//             typename InternalMatrixType::const_iterator itA = A[ k ].begin();
//             typename InternalMatrixType::const_iterator itB = B.begin();
//             typename InternalMatrixType::const_iterator itAend = A[ k ].end();
//             while ( itA != itAend )
//             {
//               matrixElementProduct += (*itA) * (*itB);
//               ++itA;
//               ++itB;
//             }
//
//             derivative[ nonZeroJacobianIndices[ mu + numParPerDim * k ] ]
//               += 2.0 * matrixElementProduct;
//           }
//         }
//       } // end if B-spline

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

#endif // #ifndef __itkAdvancedTransformRigidityPenaltyTerm_txx

