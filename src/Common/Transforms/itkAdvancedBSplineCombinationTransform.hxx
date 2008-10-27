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
AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>:: 
TransformPoint( const InputPointType & point ) const
{ 
  return this->Superclass::TransformPoint( point );

} // end TransformPoint()


/**
 * ****************** TransformPoint ****************************
 */

template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>::  
TransformPoint( const InputPointType & inputPoint,
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
void AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>
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
void AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>
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
void AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>::
TransformPointBSplineUseAddition(
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
void AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>::
TransformPointBSplineUseComposition(
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
void AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>::  
TransformPointBSplineNoInitialTransform(
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
void AdvancedBSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>::  
TransformPointBSplineNoCurrentTransform(
  const InputPointType & inputPoint,
  OutputPointType & outputPoint,
  WeightsType & weights,
  ParameterIndexArrayType & indices,
  bool & inside ) const
{
  /** Throw an exception. */
  this->NoCurrentTransformSet(); 

} // end TransformPointBSplineNoCurrentTransform()


} // end namespace itk


#endif // end #ifndef __itkAdvancedBSplineCombinationTransform_hxx

