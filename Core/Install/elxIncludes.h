/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef elxIncludes_h
#define elxIncludes_h

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
 * class ITK_TEMPLATE_EXPORT MattesMetric :
 * etc...
 */

#include "elxMacro.h"
#include "elxElastixTemplate.h"
#include "elxSupportedImageTypes.h"

/** Writing to screen and logfiles etc. */
#include "xoutmain.h"

#endif //#ifndef elxIncludes_h
