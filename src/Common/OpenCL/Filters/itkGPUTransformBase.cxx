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
#include "itkGPUTransformBase.h"

namespace itk
{
GPUTransformBase::GPUTransformBase()
{
  this->m_ParametersDataManager = GPUDataManager::New();
}


//------------------------------------------------------------------------------
bool
GPUTransformBase::GetSourceCode( std::string & /*source*/ ) const
{
  // do nothing here
  return true;
}


//------------------------------------------------------------------------------
GPUDataManager::Pointer
GPUTransformBase::GetParametersDataManager( void ) const
{
  return this->m_ParametersDataManager;
}


//------------------------------------------------------------------------------
GPUDataManager::Pointer
GPUTransformBase::GetParametersDataManager( const std::size_t index ) const
{
  return this->GetParametersDataManager();
}


} // end namespace itk
