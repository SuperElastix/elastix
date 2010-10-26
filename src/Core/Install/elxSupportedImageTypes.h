/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxSupportedImageTypes_h
#define __elxSupportedImageTypes_h

#include "elxMacro.h"

#include "elxInstallFunctions.h"
#include "elxComponentDatabase.h"
#include "elxBaseComponent.h"
#include "elxElastixTemplate.h"
#include "itkImage.h"

namespace elastix
{

  elxPrepareImageTypeSupportMacro();


  /**
   * ******************** SupportedImageTypes *********************
   *
   * Add here the combinations of ImageTypes that elastix should support.
   *
   * Syntax:
   *
   * elxSupportedImageTypeMacro( FixedImagePixelType,
   *                             FixedImageDimension,
   *                             MovingImagePixelType,
   *                             MovingImageDimension,
   *                             Index )
   *
   * Each combination of image types has as 'ID', the Index.
   * Duplicate indices are not allowed. Index 0 is not allowed.
   * The indices must form a "continuous series":
   *    ( index_{i} - index_{i-1} == 1 ).
   *
   * The NrOfSupportedImageTypes must also be set to the right value.
   *
   * elastix, and all its components, must be recompiled after adding
   * a line in this file.
   */

  const unsigned int NrOfSupportedImageTypes = 4;

//  elxSupportedImageTypeMacro( short,            2, short,             2,  1 );
  elxSupportedImageTypeMacro( short,            3, short,             3,  1 );
//  elxSupportedImageTypeMacro( char,             2, char,              2,  3 );
//  elxSupportedImageTypeMacro( char,             3, char,              3,  4 );
//  elxSupportedImageTypeMacro( int,              2, int,               2,  5 );
//  elxSupportedImageTypeMacro( int,              3, int,               3,  6 );
  elxSupportedImageTypeMacro( float,            2, float,             2,  2 );
  elxSupportedImageTypeMacro( float,            3, float,             3,  3 );
//   elxSupportedImageTypeMacro( short,            4, short,             4,  4 );
  elxSupportedImageTypeMacro( short,            4, short,             4,  4 );
//  elxSupportedImageTypeMacro( double,           2, double,            2,  9 );
//  elxSupportedImageTypeMacro( double,           3, double,            3,  10 );
//  elxSupportedImageTypeMacro( unsigned short,   2, unsigned short,    2,  11 );
//  elxSupportedImageTypeMacro( unsigned short,   3, unsigned short,    3,  12 );
//  elxSupportedImageTypeMacro( unsigned char,    2, unsigned char,     2,  5 );
//  elxSupportedImageTypeMacro( unsigned char,    3, unsigned char,     3,  4 );
//  elxSupportedImageTypeMacro( unsigned int,     2, unsigned int,      2,  15 );
//  elxSupportedImageTypeMacro( unsigned int,     3, unsigned int,      3,  16 );


  //etc


} // end namespace elastix

#endif // end #ifndef __elxSupportedImageTypes_h

