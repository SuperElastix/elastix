#ifndef __elxConfigurationToUse_h
#define __elxConfigurationToUse_h


/** 
 * The header file in which the configuration class to use
 * is declared.
 *
 * We could have also decided to allow for runtime selection
 * of the configuration class (like other elastix components,
 * which use the elxInstallMacro), but there is not much sense
 * in choosing a different configuration class at runtime (at 
 * least, we did not stumble on any situation where this would
 * be necessary.
 * 
 * \sa MyConfiguration, ConfigurationBase
 */

#include "elxMyConfiguration.h"


#endif // end #ifndef __elxConfigurationToUse_h

