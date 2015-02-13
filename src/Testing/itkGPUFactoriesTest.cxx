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
#include "itkTestHelper.h"

// Factory includes
#include "itkGPUImageFactory.h"

// Filters includes
#include "itkGPUBSplineDecompositionImageFilterFactory.h"
#include "itkGPUCastImageFilterFactory.h"
#include "itkGPURecursiveGaussianImageFilterFactory.h"
#include "itkGPUResampleImageFilterFactory.h"
#include "itkGPUShrinkImageFilterFactory.h"

// Interpolate includes
#include "itkGPULinearInterpolateImageFunctionFactory.h"
#include "itkGPUNearestNeighborInterpolateImageFunctionFactory.h"
#include "itkGPUBSplineInterpolateImageFunctionFactory.h"

// ITK transform includes
#include "itkGPUAffineTransformFactory.h"
#include "itkGPUBSplineTransformFactory.h"
#include "itkGPUCompositeTransformFactory.h"
#include "itkGPUEuler2DTransformFactory.h"
#include "itkGPUEuler3DTransformFactory.h"
#include "itkGPUIdentityTransformFactory.h"
#include "itkGPUSimilarity2DTransformFactory.h"
#include "itkGPUSimilarity3DTransformFactory.h"
#include "itkGPUTranslationTransformFactory.h"

// elastix advanced transform includes
#include "itkGPUAdvancedBSplineDeformableTransformFactory.h"
#include "itkGPUAdvancedCombinationTransformFactory.h"
//#include "itkGPUAdvancedEuler2DTransformFactory.h"
#include "itkGPUAdvancedEuler3DTransformFactory.h"
#include "itkGPUAdvancedMatrixOffsetTransformBaseFactory.h"
#include "itkGPUAdvancedSimilarity2DTransformFactory.h"
#include "itkGPUAdvancedSimilarity3DTransformFactory.h"
#include "itkGPUAdvancedTranslationTransformFactory.h"

//------------------------------------------------------------------------------
// Helper function to print all factories
void
PrintAllRegisteredFactories()
{
  // List all registered factories
  std::list< itk::ObjectFactoryBase * > factories
    = itk::ObjectFactoryBase::GetRegisteredFactories();

  std::cout << "----- Registered factories -----" << std::endl;
  for( std::list< itk::ObjectFactoryBase * >::iterator
    f = factories.begin();
    f != factories.end(); ++f )
  {
    std::cout << "  Factory version: "
              << ( *f )->GetITKSourceVersion() << std::endl
              << "  Factory description: "
              << ( *f )->GetDescription() << std::endl;

    std::list< std::string > overrides    = ( *f )->GetClassOverrideNames();
    std::list< std::string > names        = ( *f )->GetClassOverrideWithNames();
    std::list< std::string > descriptions = ( *f )->GetClassOverrideDescriptions();
    std::list< bool >        enableflags  = ( *f )->GetEnableFlags();

    std::list< std::string >::const_iterator n = names.begin();
    std::list< std::string >::const_iterator d = descriptions.begin();
    std::list< bool >::const_iterator        e = enableflags.begin();
    for( std::list< std::string >::const_iterator o = overrides.begin();
      o != overrides.end(); ++o, ++n, ++d, ++e )
    {
      std::cout << "    Override " << *o
                << " with " << *n << std::endl
                << "      described as \"" << *d << "\"" << std::endl
                << "      enabled " << *e << std::endl;
    }
  }
}


//------------------------------------------------------------------------------
// Helper function to validate factories registration
bool
IsRegistered(
  const std::size_t expectedNumberOfFactories,
  const std::size_t expectedNumberOfOverrides )
{
  std::list< itk::ObjectFactoryBase * > factories
    = itk::ObjectFactoryBase::GetRegisteredFactories();

  if( factories.size() != expectedNumberOfFactories )
  {
    std::cerr << "ERROR: Expected number of factories=" << expectedNumberOfFactories
              << ", actual=" << factories.size() << std::endl;
    return false;
  }

  std::list< std::string > overrides = factories.front()->GetClassOverrideNames();
  if( overrides.size() != expectedNumberOfOverrides )
  {
    std::cerr << "ERROR: Expected number of overrides=" << expectedNumberOfOverrides
              << ", actual=" << overrides.size() << std::endl;
    return false;
  }

  return true;
}


//------------------------------------------------------------------------------
// Definition of OCLDims for test
// We can't use OpenCLImageDimentions struct directly from the
// elxOpenCLSupportedImageTypes.h because that may change depending
// on the CMake configuration, we need it fixed for tests.
struct OCLDims
{
  itkStaticConstMacro( Support1D, bool, false );
  itkStaticConstMacro( Support2D, bool, true );
  itkStaticConstMacro( Support3D, bool, true );
};

// Define the most common elastix OpenCL image types,
// We can't use OpenCLImageTypes types directly from
// elxOpenCLSupportedImageTypes.h because they may change depending
// on CMake configuration, we need fixed list for tests.
typedef typelist::MakeTypeList< short, float >::Type OCLImageTypes;

//------------------------------------------------------------------------------
bool
TestGPUFactories()
{
  // Register one factory
  itk::GPUImageFactory2< itk::OpenCLDefaultImageTypes, itk::OpenCLDefaultImageDimentions >
  ::RegisterOneFactory();

  // Print all default registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 24 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();
  return true;
}


//------------------------------------------------------------------------------
bool
TestGPUFilterFactories()
{
  // Register one factory
  itk::GPUBSplineDecompositionImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 32 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUCastImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 48 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPURecursiveGaussianImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 32 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUResampleImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 64 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUShrinkImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 32 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();
  return true;
}


//------------------------------------------------------------------------------
bool
TestGPUInterpolatorFactories()
{
  // Register one factory
  itk::GPULinearInterpolateImageFunctionFactory2< OCLImageTypes, OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 16 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUNearestNeighborInterpolateImageFunctionFactory2< OCLImageTypes, OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 16 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUBSplineInterpolateImageFunctionFactory2< OCLImageTypes, OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 16 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  return true;
}


//------------------------------------------------------------------------------
bool
TestGPUTransformFactories()
{
  // Register one factory
  itk::GPUAffineTransformFactory2< OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 4 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUBSplineTransformFactory2< OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 12 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUCompositeTransformFactory2< OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 4 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUEuler2DTransformFactory2< OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 2 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUEuler3DTransformFactory2< OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 2 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUIdentityTransformFactory2< OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 4 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUSimilarity2DTransformFactory2< OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 2 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUSimilarity3DTransformFactory2< OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 2 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUTranslationTransformFactory2< OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 4 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();
  return true;
}


//------------------------------------------------------------------------------
bool
TestGPUAdvancedTransformFactories()
{
  // Register one factory
  itk::GPUAdvancedMatrixOffsetTransformBaseFactory2< OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 4 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUAdvancedBSplineDeformableTransformFactory2< OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 12 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUAdvancedCombinationTransformFactory2< OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 4 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  //itk::GPUAdvancedEuler2DTransformFactory< OCLImageDims2 >
  //::RegisterOneFactory();

  //// Print all elastix registered factories
  //PrintAllRegisteredFactories();
  //if( !IsRegistered( 1, 2 ) )
  //{
  //  return false;
  //}

  //// Unregister all
  //itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUAdvancedEuler3DTransformFactory2< OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 2 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUAdvancedSimilarity2DTransformFactory2< OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 2 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUAdvancedSimilarity3DTransformFactory2< OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 2 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  // Register one factory
  itk::GPUAdvancedTranslationTransformFactory2< OCLDims >
  ::RegisterOneFactory();

  // Print all elastix registered factories
  PrintAllRegisteredFactories();
  if( !IsRegistered( 1, 4 ) )
  {
    return false;
  }

  // Unregister all
  itk::ObjectFactoryBase::UnRegisterAllFactories();
  return true;
}


//------------------------------------------------------------------------------
// This test validates GPU factory create process
int
main( int argc, char * argv[] )
{
  // Setup for debugging
  itk::SetupForDebugging();

  // Create and check OpenCL context
  if( !itk::CreateContext() )
  {
    return EXIT_FAILURE;
  }

  // ITK creates some factories unregister them
  itk::ObjectFactoryBase::UnRegisterAllFactories();

  if( !TestGPUFactories() )
  {
    return EXIT_FAILURE;
  }

  if( !TestGPUFilterFactories() )
  {
    return EXIT_FAILURE;
  }

  if( !TestGPUInterpolatorFactories() )
  {
    return EXIT_FAILURE;
  }

  if( !TestGPUTransformFactories() )
  {
    return EXIT_FAILURE;
  }

  if( !TestGPUAdvancedTransformFactories() )
  {
    return EXIT_FAILURE;
  }

  // End program.
  return EXIT_SUCCESS;
} // end main()
