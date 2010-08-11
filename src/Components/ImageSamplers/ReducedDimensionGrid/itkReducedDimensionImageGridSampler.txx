/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __ReducedDimensionImageGridSampler_txx
#define __ReducedDimensionImageGridSampler_txx

#include "itkReducedDimensionImageGridSampler.h"
#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{

  /**
  * ******************* SetInputImageRegion *******************
  */

  template< class TInputImage >
  void
    ReducedDimensionImageGridSampler< TInputImage >
    ::SetInputImageRegion( const InputImageRegionType _arg, unsigned int pos )
  {
    /** Reduce image region in given dimension. */
    InputImageRegionType newRegion = _arg;
    newRegion.SetSize( this->m_ReducedDimension, 1 );
    newRegion.SetIndex( this->m_ReducedDimension, this->m_ReducedDimensionIndex );

    /** Call superclass to set the new image region. */
    Superclass::SetInputImageRegion( newRegion, pos );
  } // SetInputImageRegion()


  /**
   * ******************* PrintSelf *******************
   */

  template< class TInputImage >
    void
    ReducedDimensionImageGridSampler< TInputImage >
    ::PrintSelf( std::ostream& os, Indent indent ) const
  {
    Superclass::PrintSelf( os, indent );
  } // end PrintSelf


} // end namespace itk

#endif // end #ifndef __ReducedDimensionImageGridSampler_txx

