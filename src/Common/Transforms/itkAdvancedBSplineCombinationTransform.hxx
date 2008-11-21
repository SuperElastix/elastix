/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkAdvancedBSplineCombinationTransform_hxx
#define __itkAdvancedBSplineCombinationTransform_hxx

#include "itkAdvancedBSplineCombinationTransform.h"


namespace itk
{
  
/**
 * ************************ Constructor *************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>
::AdvancedBSplineCombinationTransform() : Superclass()
{
  /** Initialize.*/
  this->m_CurrentTransformAsBSplineTransform = 0;

  this->m_SelectedTransformPointBSplineFunction
    = &Self::TransformPointBSplineNoCurrentTransform;

} // end Constructor


/**
 * ****************** TransformPoint ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
typename AdvancedBSplineCombinationTransform<
TScalarType, NDimensions, VSplineOrder>::OutputPointType
AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>
::TransformPoint( const InputPointType & point ) const
{ 
  return this->Superclass::TransformPoint( point );

} // end TransformPoint()


/**
 * ****************** TransformPoint ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>
::TransformPoint( const InputPointType & inputPoint,
  OutputPointType & outputPoint,
  WeightsType & weights,
  ParameterIndexArrayType & indices,
  bool & inside ) const
{
  /** Call the selected TransformPointBSplineFunction. */
  ((*this).*m_SelectedTransformPointBSplineFunction)(
    inputPoint, outputPoint, weights, indices, inside );

} // end TransformPoint with extra arguments


/**
 * ******************* SetCurrentTransform **********************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>
::SetCurrentTransform( CurrentTransformType * _arg )
{
  /** Set the the current transform and call the UpdateCombinationMethod. */
  if ( this->m_CurrentTransform != _arg )
  {
    /** If a zero pointer is given: */
    if ( _arg == 0 )
    {
      this->m_CurrentTransform = 0;
      this->m_CurrentTransformAsBSplineTransform = 0;
      this->Modified();
      this->UpdateCombinationMethod();
      return;
    }

    /** If the pointer is nonzero, try to cast it to a BSpline transform. */
    BSplineTransformType * testPointer = 
      dynamic_cast<BSplineTransformType *>( _arg );
    if ( testPointer )
    {
      this->m_CurrentTransform = _arg;
      this->m_CurrentTransformAsBSplineTransform = testPointer;
      this->Modified();
      this->UpdateCombinationMethod();
    }
    else
    {
      itkExceptionMacro( << "The entered CurrentTransform is not a BSplineTransform." );
    }
  }

} // end SetCurrentTransform()


/**
 * ****************** UpdateCombinationMethod ********************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>
::UpdateCombinationMethod( void )
{
  this->Superclass::UpdateCombinationMethod();

  /** Update the m_SelectedTransformPointBSplineFunction. */
  if ( this->m_CurrentTransform.IsNull() )
  {
    this->m_SelectedTransformPointBSplineFunction
      = &Self::TransformPointBSplineNoCurrentTransform;
  }
  else if ( this->m_InitialTransform.IsNull() )
  {
    this->m_SelectedTransformPointBSplineFunction
      = &Self::TransformPointBSplineNoInitialTransform;
  }
  else if ( this->m_UseAddition )
  {
    this->m_SelectedTransformPointBSplineFunction
      = &Self::TransformPointBSplineUseAddition;
  }
  else
  {
    this->m_SelectedTransformPointBSplineFunction
      = &Self::TransformPointBSplineUseComposition;
  }

} // end UpdateCombinationMethod()


/**
 * ************* TransformPointBSplineUseAddition **********************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>
::TransformPointBSplineUseAddition(
  const InputPointType & inputPoint,
  OutputPointType & outputPoint,
  WeightsType & weights,
  ParameterIndexArrayType & indices,
  bool & inside ) const
{       
  /** The Initial transform. */
  OutputPointType out0
    = this->m_InitialTransform->TransformPoint( inputPoint );

  /** The Current transform. */
  this->m_CurrentTransformAsBSplineTransform->TransformPoint( 
    inputPoint, outputPoint, weights, indices, inside );

  /** Both added together. */
  for ( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    outputPoint[ i ] += ( out0[ i ] - inputPoint[ i ] );
  }

} // end TransformPointBSplineUseAddition()


/**
 * **************** TransformPointBSplineUseComposition *************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>
::TransformPointBSplineUseComposition(
  const InputPointType & inputPoint,
  OutputPointType & outputPoint,
  WeightsType & weights,
  ParameterIndexArrayType & indices, 
  bool & inside ) const
{
  this->m_CurrentTransformAsBSplineTransform->TransformPoint( 
    this->m_InitialTransform->TransformPoint( inputPoint ),
    outputPoint, weights, indices, inside );

} // end TransformPointBSplineUseComposition()


/**
 * **************** TransformPointBSplineNoInitialTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>
::TransformPointBSplineNoInitialTransform(
  const InputPointType & inputPoint,
  OutputPointType & outputPoint,
  WeightsType & weights,
  ParameterIndexArrayType & indices,
  bool & inside ) const
{
  this->m_CurrentTransformAsBSplineTransform->TransformPoint(
    inputPoint, outputPoint, weights, indices, inside );

} // end TransformPointBSplineNoInitialTransform()


/**
 * ******** TransformPointBSplineNoCurrentTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>
::TransformPointBSplineNoCurrentTransform(
  const InputPointType & inputPoint,
  OutputPointType & outputPoint,
  WeightsType & weights,
  ParameterIndexArrayType & indices,
  bool & inside ) const
{
  /** Throw an exception. */
  this->NoCurrentTransformSet(); 

} // end TransformPointBSplineNoCurrentTransform()


/**
 * ******** GetJacobianOfSpatialHessianUseComposition ******************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>
::GetJacobianOfSpatialHessianUseComposition(
  const InputPointType & ipp,
  SpatialHessianType & sh,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  /** Create intermediary variables for the internal transforms. */
  SpatialJacobianType sj0, sj1;
  SpatialHessianType sh0, sh1;
  JacobianOfSpatialJacobianType jsj1;
  JacobianOfSpatialHessianType jsh1;

  unsigned long numberOfNZJI = jsh.size();
  jsj1.resize( numberOfNZJI );
  jsh1.resize( numberOfNZJI );

  /** Transform the input point. */
  // \todo this has already been computed and it is expensive.
  InputPointType transformedPoint
    = this->m_InitialTransform->TransformPoint( ipp );

  /** Compute the (Jacobian of the) spatial Jacobian / Hessian of the
   * internal transforms.
   */
  this->m_InitialTransform->GetSpatialJacobian( ipp, sj0 );
  this->m_InitialTransform->GetSpatialHessian( ipp, sh0 );
  this->m_CurrentTransform->GetJacobianOfSpatialJacobian(
    transformedPoint, sj1, jsj1, nonZeroJacobianIndices );
  this->m_CurrentTransform->GetJacobianOfSpatialHessian(
    transformedPoint, sh1, jsh1, nonZeroJacobianIndices );

  /** Combine them in one overall spatial Hessian. */
  for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
  {
    for ( unsigned int i = 0; i < SpaceDimension; ++i )
    {
      for ( unsigned int j = 0; j < SpaceDimension; ++j )
      {
        sh[ dim ]( i, j )
          = sj0( dim, j ) * sh1[ dim ]( i, j )
          + sh0[ dim ]( i, j ) * sj1( dim, j );
      }
    }
  }

  /** Combine them in one overall Jacobian of spatial Hessian. */
  unsigned int numParPerDim
    = nonZeroJacobianIndices.size() / SpaceDimension;
  for ( unsigned int mu = 0; mu < numParPerDim; ++mu )
  {
    SpatialJacobianType matrix = jsh1[ mu ][ 0 ];
    for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
    {
      SpatialJacobianType matrix2;
      for ( unsigned int i = 0; i < SpaceDimension; ++i )
      {
        for ( unsigned int j = 0; j < SpaceDimension; ++j )
        {
          matrix2( i, j ) = sj0( dim, j ) * matrix( i, j );
        }
      }
      jsh[ mu + numParPerDim * dim ][ dim ] = matrix2;
    }
  }

  for ( unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu )
  {
    SpatialJacobianType matrix = jsj1[ mu ];
    for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
    {
      for ( unsigned int i = 0; i < SpaceDimension; ++i )
      {
        jsh[ mu ][ dim ]( dim, i ) += sh0[ dim ]( dim, i ) * matrix( dim, i );
      }
    }
  }

} // end GetJacobianOfSpatialHessianUseComposition()


} // end namespace itk


#endif // end #ifndef __itkAdvancedBSplineCombinationTransform_hxx

