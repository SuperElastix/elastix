/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxIncludes_h
#define __elxIncludes_h

/**
 * Include this file in your new components.
 *
 * example:
 *
 * // elxMattesMetric.h //
 * #include "elxIncludes.h"
 * #include "itkMattesMutualInformationImageToImageMetric.h"
 *
 * namespace elx
 * {
 *
 * template <class TElastix>
 * class MattesMetric :
 * etc...
 */

#include "elxMacro.h"
#include "elxElastixTemplate.h"
#include "elxSupportedImageTypes.h"

/** Writing to screen and logfiles etc. */
#include "xoutmain.h"

#endif //#ifndef __elxIncludes_h
