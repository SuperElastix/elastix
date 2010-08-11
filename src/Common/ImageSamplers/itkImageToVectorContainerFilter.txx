/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __ImageToVectorContainerFilter_txx
#define __ImageToVectorContainerFilter_txx

#include "itkImageToVectorContainerFilter.h"

namespace itk
{

  /**
   * ******************* Constructor *******************
   */

  template< class TInputImage, class TOutputVectorContainer >
    ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
    ::ImageToVectorContainerFilter()
  {
    this->ProcessObject::SetNumberOfRequiredInputs( 1 );

    OutputVectorContainerPointer output
      = dynamic_cast<OutputVectorContainerType*>( this->MakeOutput(0).GetPointer() );

    this->ProcessObject::SetNumberOfRequiredOutputs( 1 );
    this->ProcessObject::SetNthOutput( 0, output.GetPointer() );
  } // end Constructor


  /**
   * ******************* MakeOutput *******************
   */

  template< class TInputImage, class TOutputVectorContainer >
    DataObject::Pointer
    ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
    ::MakeOutput( unsigned int idx )
  {
    OutputVectorContainerPointer outputVectorContainer = OutputVectorContainerType::New();
    return dynamic_cast< DataObject * >( outputVectorContainer.GetPointer() );
  } // end MakeOutput


  /**
   * ******************* SetInput *******************
   */

  template< class TInputImage, class TOutputVectorContainer >
    void
    ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
    ::SetInput( unsigned int idx, const InputImageType *input )
  {
    // process object is not const-correct, the const_cast
    // is required here.
    this->ProcessObject::SetNthInput( idx,
      const_cast< InputImageType * >( input ) );
  } // end SetInput


  /**
   * ******************* SetInput *******************
   */

  template< class TInputImage, class TOutputVectorContainer >
    void
    ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
    ::SetInput( const InputImageType *input )
  {
    this->ProcessObject::SetNthInput( 0, const_cast< InputImageType * >( input ) );
  } // end SetInput


  /**
   * ******************* GetInput *******************
   */

  template< class TInputImage, class TOutputVectorContainer >
    const typename ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >::InputImageType *
    ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
    ::GetInput( void )
  {
    return dynamic_cast< const InputImageType * >(
      this->ProcessObject::GetInput( 0 ) );
  } // end GetInput


  /**
   * ******************* GetInput *******************
   */

  template< class TInputImage, class TOutputVectorContainer >
    const typename ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >::InputImageType *
    ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
    ::GetInput( unsigned int idx )
  {
    return dynamic_cast< const InputImageType * >(
      this->ProcessObject::GetInput( idx ) );
  } // end GetInput


  /**
   * ******************* GetOutput *******************
   */

  template< class TInputImage, class TOutputVectorContainer >
    typename ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >::OutputVectorContainerType *
    ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
    ::GetOutput( void )
  {
    return dynamic_cast< OutputVectorContainerType * >(
      this->ProcessObject::GetOutput( 0 ) );
  } // end GetOutput


  /**
   * ******************* PrintSelf *******************
   */

  template< class TInputImage, class TOutputVectorContainer >
    void
    ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
    ::PrintSelf( std::ostream& os, Indent indent ) const
  {
    Superclass::PrintSelf( os, indent );
  } // end PrintSelf


  /**
   * ******************* GenerateOutputInformation *******************
   *
   * Copy information from first input to all outputs.
   * This is a void implementation to prevent the
   * ProcessObject version to be called.
   *

  template< class TInputImage, class TOutputVectorContainer >
    void
    ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
    ::GenerateOutputInformation( void )
  {
  } // end GenerateOutputInformation
*/

} // end namespace itk

#endif // end #ifndef __ImageToVectorContainerFilter_txx

