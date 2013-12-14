/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxWeightedCombinationTransform_HXX_
#define __elxWeightedCombinationTransform_HXX_

#include "elxWeightedCombinationTransform.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template< class TElastix >
WeightedCombinationTransformElastix< TElastix >
::WeightedCombinationTransformElastix()
{
  this->m_WeightedCombinationTransform
    = WeightedCombinationTransformType::New();
  this->SetCurrentTransform( this->m_WeightedCombinationTransform );
}   // end Constructor


/*
 * ******************* BeforeRegistration ***********************
 */

template< class TElastix >
void
WeightedCombinationTransformElastix< TElastix >
::BeforeRegistration( void )
{
  /** Set the normalizedWeights parameter. It must be correct in order to set the scales properly.
   * \todo: this parameter may change each resolution. */
  bool normalizeWeights = false;
  this->m_Configuration->ReadParameter( normalizeWeights,
    "NormalizeCombinationWeights", 0 );
  this->m_WeightedCombinationTransform->SetNormalizeWeights( normalizeWeights );

  /** Give initial parameters to this->m_Registration.*/
  this->InitializeTransform();

  /** Set the scales. */
  this->SetScales();

}   // end BeforeRegistration


/**
 * ************************* InitializeTransform *********************
 * Initialize transform to prepare it for registration.
 */

template< class TElastix >
void
WeightedCombinationTransformElastix< TElastix >
::InitializeTransform( void )
{
  /** Load subtransforms specified in parameter file. */
  this->LoadSubTransforms();

  /** Some helper variables */
  const NumberOfParametersType N  = this->GetNumberOfParameters();
  const double                 Nd = static_cast< double >( N );

  /** Equal weights */
  ParametersType parameters( N );
  if( this->m_WeightedCombinationTransform->GetNormalizeWeights() )
  {
    parameters.Fill( 1.0 / Nd );
  }
  else
  {
    parameters.Fill( 0.0 );
  }
  this->m_WeightedCombinationTransform->SetParameters( parameters );

  /** Set the initial parameters in this->m_Registration.*/
  this->m_Registration->GetAsITKBaseType()->
  SetInitialTransformParameters( this->GetParameters() );

}   // end InitializeTransform


/**
 * ************************* ReadFromFile ************************
 */

template< class TElastix >
void
WeightedCombinationTransformElastix< TElastix >
::ReadFromFile( void )
{
  /** Load subtransforms specified in transform parameter file. */
  this->LoadSubTransforms();

  /** Set the normalizeWeights option */
  bool normalizeWeights = false;
  this->m_Configuration->ReadParameter( normalizeWeights,
    "NormalizeCombinationWeights", 0 );
  this->m_WeightedCombinationTransform->SetNormalizeWeights( normalizeWeights );

  /** Call the ReadFromFile from the TransformBase to read in the parameters.  */
  this->Superclass2::ReadFromFile();

}   // end ReadFromFile()


/**
* ************************* WriteToFile ************************
*/

template< class TElastix >
void
WeightedCombinationTransformElastix< TElastix >
::WriteToFile( const ParametersType & param ) const
{
  /** Call the WriteToFile from the TransformBase. */
  this->Superclass2::WriteToFile( param );

  /** Write WeightedCombinationTransform specific things. */
  xout[ "transpar" ] << std::endl << "// WeightedCombinationTransform specific" << std::endl;

  /** Write normalize-weights option */
  std::string normalizeString = "false";
  if( this->m_WeightedCombinationTransform->GetNormalizeWeights() )
  {
    normalizeString = "true";
  }
  xout[ "transpar" ] << "(NormalizeCombinationWeights \"" << normalizeString << "\" )" << std::endl;

  /** Write names of subtransforms */
  xout[ "transpar" ] << "(SubTransforms ";
  for( unsigned int i = 0; i < this->m_SubTransformFileNames.size(); ++i )
  {
    xout[ "transpar" ] << "\"" << this->m_SubTransformFileNames[ i ] << "\" ";
  }
  xout[ "transpar" ] << ")" << std::endl;

}   // end WriteToFile()


/**
* ************************* SetScales *********************
*/

template< class TElastix >
void
WeightedCombinationTransformElastix< TElastix >
::SetScales( void )
{
  /** Create the new scales. */
  const NumberOfParametersType N = this->GetNumberOfParameters();
  ScalesType                   newscales( N );
  newscales.Fill( 1.0 );

  /** Check if automatic scales estimation is desired. */
  bool automaticScalesEstimation = false;
  this->m_Configuration->ReadParameter( automaticScalesEstimation,
    "AutomaticScalesEstimation", 0, false );

  if( automaticScalesEstimation )
  {
    elxout << "Scales are estimated automatically." << std::endl;
    this->AutomaticScalesEstimation( newscales );
  }
  else
  {
    const std::size_t count
      = this->m_Configuration->CountNumberOfParameterEntries( "Scales" );

    if( count == N )
    {
      /** Read the user-supplied values/ */
      std::vector< double > newscalesvec( N );
      this->m_Configuration->ReadParameter( newscalesvec, "Scales", 0, N - 1, true );
      for( unsigned int i = 0; i < N; i++ )
      {
        newscales[ i ] = newscalesvec[ i ];
      }
    }
    else if( count != 0 )
    {
      /** In this case an error is made in the parameter-file.
      * An error is thrown, because using erroneous scales in the optimizer
      * can give unpredictable results.
      */
      itkExceptionMacro( << "ERROR: The Scales-option in the parameter-file"
                         << " has not been set properly." );
    }

  }   // end else: no automaticScalesEstimation

  elxout << "Scales for transform parameters are: " << newscales << std::endl;

  /** And set the scales into the optimizer. */
  this->m_Registration->GetAsITKBaseType()->GetOptimizer()->SetScales( newscales );

}   // end SetScales()


/**
* ************************* LoadSubTransforms *********************
*/

template< class TElastix >
void
WeightedCombinationTransformElastix< TElastix >
::LoadSubTransforms( void )
{
  /** Typedef's from ComponentDatabase. */
  typedef typename Superclass2::ComponentDatabaseType    ComponentDatabaseType;
  typedef typename Superclass2::ComponentDescriptionType ComponentDescriptionType;
  typedef typename Superclass2::PtrToCreator             PtrToCreator;
  typedef typename Superclass2::ObjectType               ObjectType;

  const std::size_t N
    = this->m_Configuration->CountNumberOfParameterEntries( "SubTransforms" );

  if( N == 0 )
  {
    itkExceptionMacro( << "ERROR: At least one SubTransform should be specified." );
  }
  else
  {
    this->m_SubTransformFileNames.resize( N );
    this->m_Configuration->ReadParameter( this->m_SubTransformFileNames,
      "SubTransforms", 0, N - 1, true );
  }

  /** Create a vector of subTransform pointers and initialize to null pointers.
   * \todo: make it a member variable if it appears to needed later */
  TransformContainerType subTransforms( N, 0 );

  /** Load each subTransform */
  for( unsigned int i = 0; i < N; ++i )
  {
    /** \todo: large parts of these code were copied from the elx::TransformBase.
     * Could we put some functionality in a function? */

    /** Read the name of the subTransform */
    const std::string & subTransformFileName = this->m_SubTransformFileNames[ i ];

    /** Create a new configuration, which will be initialized with
     * the subtransformFileName. */
    ConfigurationPointer configurationSubTransform = ConfigurationType::New();

    /** Create argmapInitialTransform. */
    CommandLineArgumentMapType argmapSubTransform;
    argmapSubTransform.insert( CommandLineEntryType(
      "-tp", subTransformFileName ) );

    int initfailure = configurationSubTransform->Initialize( argmapSubTransform );
    if( initfailure != 0 )
    {
      itkExceptionMacro( << "ERROR: Reading SubTransform "
                         << "parameters failed: " << subTransformFileName );
    }

    /** Read the SubTransform name. */
    ComponentDescriptionType subTransformName = "AffineTransform";
    configurationSubTransform->ReadParameter(
      subTransformName, "Transform", 0 );

    /** Create a SubTransform. */
    typename ObjectType::Pointer subTransform;
    PtrToCreator testcreator = 0;
    testcreator = this->GetElastix()->GetComponentDatabase()
      ->GetCreator( subTransformName, this->m_Elastix->GetDBIndex() );
    subTransform = testcreator ? testcreator() : NULL;

    /** Cast to TransformBase */
    Superclass2 * elx_subTransform = dynamic_cast< Superclass2 * >(
      subTransform.GetPointer() );

    /** Call the ReadFromFile method of the elx_subTransform. */
    if( elx_subTransform )
    {
      elx_subTransform->SetElastix( this->GetElastix() );
      elx_subTransform->SetConfiguration( configurationSubTransform );
      elx_subTransform->ReadFromFile();

      /** Set in vector of subTransforms. */
      SubTransformType * testPointer
                         = dynamic_cast< SubTransformType * >( subTransform.GetPointer() );
      subTransforms[ i ] = testPointer;
    }

    /** Check if no errors occured: */
    if( subTransforms[ i ].IsNull() )
    {
      xl::xout[ "error" ] << "ERROR: Error while trying to load the SubTransform "
                          << subTransformFileName << std::endl;
      itkExceptionMacro( << "ERROR: Loading SubTransforms failed!" );
    }

  }    // end for loop over subTransforms

  /** Set the subTransforms in the WeightedCombination object. */
  this->m_WeightedCombinationTransform->SetTransformContainer( subTransforms );

}   // end LoadSubTransforms()


} // end namespace elastix

#endif // end #ifndef __elxWeightedCombinationTransform_HXX_
