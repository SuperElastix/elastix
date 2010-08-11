/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __VectorContainerSource_txx
#define __VectorContainerSource_txx

#include "itkVectorContainerSource.h"

namespace itk
{

  /**
   * ******************* Constructor *******************
   */

  template< class TOutputVectorContainer >
    VectorContainerSource< TOutputVectorContainer >
    ::VectorContainerSource()
  {
    // Create the output. We use static_cast<> here because we know the default
    // output must be of type TOutputVectorContainer
    OutputVectorContainerPointer output
      = static_cast< TOutputVectorContainer * >( this->MakeOutput(0).GetPointer() );

    this->ProcessObject::SetNumberOfRequiredOutputs( 1 );
    this->ProcessObject::SetNthOutput( 0, output.GetPointer() );

    m_GenerateDataRegion = 0;
    m_GenerateDataNumberOfRegions = 0;
  } // end Constructor


  /**
   * ******************* MakeOutput *******************
   */

  template< class TOutputVectorContainer >
    typename VectorContainerSource< TOutputVectorContainer >::DataObjectPointer
    VectorContainerSource< TOutputVectorContainer >
    ::MakeOutput( unsigned int idx )
  {
    return static_cast<DataObject*>( TOutputVectorContainer::New().GetPointer() );
  } // end MakeOutput


  /**
   * ******************* GetOutput *******************
   */

  template< class TOutputVectorContainer >
    typename VectorContainerSource< TOutputVectorContainer >::OutputVectorContainerType *
    VectorContainerSource< TOutputVectorContainer >
    ::GetOutput( void )
  {
    if ( this->GetNumberOfOutputs() < 1 )
    {
      return 0;
    }

    return static_cast<OutputVectorContainerType *>(
      this->ProcessObject::GetOutput(0) );
  } // end GetOutput


  /**
   * ******************* GetOutput *******************
   */

  template< class TOutputVectorContainer >
    typename VectorContainerSource< TOutputVectorContainer >::OutputVectorContainerType *
    VectorContainerSource< TOutputVectorContainer >
    ::GetOutput( unsigned int idx )
  {
    return static_cast<OutputVectorContainerType *>(
      this->ProcessObject::GetOutput( idx ) );
  } // end GetOutput


  /**
   * ******************* GenerateInputRequestedRegion *******************
   */

  template< class TOutputVectorContainer >
    void
    VectorContainerSource< TOutputVectorContainer >
    ::GenerateInputRequestedRegion( void )
  {
    Superclass::GenerateInputRequestedRegion();
  } // end GenerateInputRequestedRegion


  /**
   * ******************* GraftOutput *******************
   */

  template< class TOutputVectorContainer >
    void
    VectorContainerSource< TOutputVectorContainer >
    ::GraftOutput( DataObject *graft )
  {
    this->GraftNthOutput( 0, graft );
  } // end GraftOutput


  /**
   * ******************* GraftNthOutput *******************
   */

  template< class TOutputVectorContainer >
    void
    VectorContainerSource< TOutputVectorContainer >
    ::GraftNthOutput( unsigned int idx, DataObject *graft )
  {
    /** Check idx. */
    if ( idx >= this->GetNumberOfOutputs() )
    {
      itkExceptionMacro( << "Requested to graft output " << idx
        << " but this filter only has " << this->GetNumberOfOutputs() << " Outputs." );
    }

    /** Check graft. */
    if ( !graft )
    {
      itkExceptionMacro( << "Requested to graft output that is a NULL pointer" );
    }

    /** Get a pointer to the output. */
    DataObject * output = this->GetOutput( idx );

    /** Call Graft on the vector container in order to
     * copy meta-information, and containers. */
    output->Graft( graft );

  } // end GraftNthOutput


  /**
   * ******************* PrintSelf *******************
   */

  template< class TOutputVectorContainer >
    void
    VectorContainerSource< TOutputVectorContainer >
    ::PrintSelf( std::ostream& os, Indent indent ) const
  {
    Superclass::PrintSelf( os, indent );
  } // end PrintSelf


} // end namespace itk

#endif // end #ifndef __VectorContainerSource_txx

