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

#ifndef elxComponentInstaller_h
#define elxComponentInstaller_h

#include "elxInstallFunctions.h"
#include "elxPrepareImageTypeSupport.h" // For ElastixTypedef.
#include "elxSupportedImageTypes.h"     // For NrOfSupportedImageTypes.

namespace elastix
{
/** Installs the component specified by its first argument. Note that `TComponent` is a "template template parameter".
 * It has a `TElastix` as template parameter, for example `elastix::TranslationTransformElastix` or
 * `elastix::FixedSmoothingPyramid`. */
template <template <class TElastix> class TComponent, unsigned VIndex = 1>
class ComponentInstaller
{
public:
  static int
  DO(ComponentDatabase * cdb)
  {
    using ElastixType = typename ElastixTypedef<VIndex>::ElastixType;
    const auto name = TComponent<ElastixType>::elxGetClassNameStatic();
    const int  dummy = InstallFunctions<TComponent<ElastixType>>::InstallComponent(name, VIndex, cdb);
    if (ElastixTypedef<VIndex + 1>::IsDefined)
    {
      return ComponentInstaller<TComponent, VIndex + 1>::DO(cdb);
    }
    return dummy;
  }
};


template <template <class TElastix> class TComponent>
class ComponentInstaller<TComponent, NrOfSupportedImageTypes + 1>
{
public:
  static int
  DO(ComponentDatabase * /** cdb */)
  {
    return 0;
  }
};

} // namespace elastix

#endif
