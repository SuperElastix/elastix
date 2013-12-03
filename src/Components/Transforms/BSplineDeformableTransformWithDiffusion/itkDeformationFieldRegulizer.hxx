/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkDeformationFieldRegulizer_HXX__
#define __itkDeformationFieldRegulizer_HXX__

#include "itkDeformationFieldRegulizer.h"

namespace itk
{

/**
 * ************************ Constructor *************************
 */

template< class TAnyITKTransform >
DeformationFieldRegulizer< TAnyITKTransform >
::DeformationFieldRegulizer()
{
  /** Initialize. */
  this->m_IntermediaryDeformationFieldTransform = 0;
  this->m_Initialized                           = false;

} // end Constructor


/**
 * ********* InitializeIntermediaryDeformationField **************
 */

template< class TAnyITKTransform >
void
DeformationFieldRegulizer< TAnyITKTransform >
::InitializeDeformationFields( void )
{
  /** Initialize this->m_IntermediaryDeformationFieldTransform. */
  this->m_IntermediaryDeformationFieldTransform = IntermediaryDFTransformType::New();

  /** Initialize this->m_IntermediaryDeformationField. */
  typename VectorImageType::Pointer intermediaryDeformationField = VectorImageType::New();
  intermediaryDeformationField->SetRegions( this->m_DeformationFieldRegion );
  intermediaryDeformationField->SetSpacing( this->m_DeformationFieldSpacing );
  intermediaryDeformationField->SetOrigin( this->m_DeformationFieldOrigin );
  try
  {
    intermediaryDeformationField->Allocate();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception and throw again. */
    excp.SetLocation( "DeformationFieldRegulizer - InitializeDeformationFields()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while allocating the intermediary deformation field.\n";
    excp.SetDescription( err_str );
    throw excp;
  }

  /** Set everything to zero. */
  IteratorType it( intermediaryDeformationField,
  intermediaryDeformationField->GetLargestPossibleRegion() );
  VectorPixelType vec;
  vec.Fill( NumericTraits< ScalarType >::Zero );
  while( !it.IsAtEnd() )
  {
    it.Set( vec );
    ++it;
  }

  /** Set the deformation field in the transform. */
  this->m_IntermediaryDeformationFieldTransform
  ->SetCoefficientVectorImage( intermediaryDeformationField );

  /** Set to initialized. */
  this->m_Initialized = true;

} // end InitializeDeformationFields()


/**
 * *********************** TransformPoint ***********************
 */

template< class TAnyITKTransform >
typename DeformationFieldRegulizer< TAnyITKTransform >::OutputPointType
DeformationFieldRegulizer< TAnyITKTransform >
::TransformPoint( const InputPointType & inputPoint ) const
{
  /** Get the outputpoint of any ITK Transform and the deformation field. */
  OutputPointType oppAnyT, oppDF, opp;
  oppAnyT = this->Superclass::TransformPoint( inputPoint );
  oppDF   = this->m_IntermediaryDeformationFieldTransform->TransformPoint( inputPoint );

  /** Add them: don't forget to subtract ipp. */
  for( unsigned int i = 0; i < OutputSpaceDimension; i++ )
  {
    opp[ i ] = oppAnyT[ i ] + oppDF[ i ] - inputPoint[ i ];
  }

  /** Return a value. */
  return opp;

} // end TransformPoint()


/**
 * ******** UpdateIntermediaryDeformationFieldTransform *********
 */

template< class TAnyITKTransform >
void
DeformationFieldRegulizer< TAnyITKTransform >
::UpdateIntermediaryDeformationFieldTransform(
  typename VectorImageType::Pointer vecImage )
{
  /** Set the vecImage (which is allocated elsewhere) and put it in
   * IntermediaryDeformationFieldTransform (where it is copied and split up).
   */
  this->m_IntermediaryDeformationFieldTransform
  ->SetCoefficientVectorImage( vecImage );

} // end UpdateIntermediaryDeformationFieldTransform()


} // end namespace itk

#endif // end #ifndef __itkDeformationFieldRegulizer_HXX__
