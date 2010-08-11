/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxResampleInterpolatorBase_hxx
#define __elxResampleInterpolatorBase_hxx

#include "elxResampleInterpolatorBase.h"

namespace elastix
{
using namespace itk;


/**
 * ******************* ReadFromFile *****************************
 */

template <class TElastix>
void
ResampleInterpolatorBase<TElastix>
::ReadFromFile( void )
{
  // nothing, but must be here

} // end ReadFromFile


/**
 * ******************* WriteToFile ******************************
 */

template <class TElastix>
void
ResampleInterpolatorBase<TElastix>
::WriteToFile( void ) const
{
  /** Write ResampleInterpolator specific things. */
  xl::xout["transpar"] << "\n// ResampleInterpolator specific" << std::endl;

  /** Write the name of the resample-interpolator. */
  xl::xout["transpar"] << "(ResampleInterpolator \""
    << this->elxGetClassName() << "\")" << std::endl;

} // end WriteToFile()


} // end namespace elastix

#endif // end #ifndef __elxResampleInterpolatorBase_hxx

