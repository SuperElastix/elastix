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
#ifndef itkGPUCompositeTransformCopier_hxx
#define itkGPUCompositeTransformCopier_hxx

#include "itkGPUCompositeTransformCopier.h"

namespace itk
{
//------------------------------------------------------------------------------
template <typename TTypeList,
          typename NDimensions,
          typename TCompositeTransform,
          typename TOutputTransformPrecisionType>
GPUCompositeTransformCopier<TTypeList, NDimensions, TCompositeTransform, TOutputTransformPrecisionType>::
  GPUCompositeTransformCopier()
{
  this->m_InputTransform = nullptr;
  this->m_Output = nullptr;
  this->m_InternalTransformTime = 0;
  this->m_ExplicitMode = true;
  this->m_TransformCopier = GPUTransformCopierType::New();
}


//------------------------------------------------------------------------------
template <typename TTypeList,
          typename NDimensions,
          typename TCompositeTransform,
          typename TOutputTransformPrecisionType>
void
GPUCompositeTransformCopier<TTypeList, NDimensions, TCompositeTransform, TOutputTransformPrecisionType>::Update()
{
  if (!this->m_InputTransform)
  {
    itkExceptionMacro(<< "Input CompositeTransform has not been connected");
    return;
  }

  // Update only if the input AdvancedCombinationTransform has been modified
  const ModifiedTimeType t = this->m_InputTransform->GetMTime();

  if (t == this->m_InternalTransformTime)
  {
    return; // No need to update
  }
  else if (t > this->m_InternalTransformTime)
  {
    // Cache the timestamp
    this->m_InternalTransformTime = t;

    // Create the output
    this->m_Output = GPUCompositeTransformType::New();

    // Set the same explicit mode
    this->m_TransformCopier->SetExplicitMode(this->m_ExplicitMode);

    for (std::size_t i = 0; i < m_InputTransform->GetNumberOfTransforms(); ++i)
    {
      const CPUTransformPointer fromTransform = this->m_InputTransform->GetNthTransform(i);

      // Perform copy
      this->m_TransformCopier->SetInputTransform(fromTransform);
      this->m_TransformCopier->Update();
      GPUOutputTransformPointer toTransform = this->m_TransformCopier->GetModifiableOutput();

      // Add to output
      this->m_Output->AddTransform(toTransform);
    }
  }
}


//------------------------------------------------------------------------------
template <typename TTypeList,
          typename NDimensions,
          typename TCompositeTransform,
          typename TOutputTransformPrecisionType>
void
GPUCompositeTransformCopier<TTypeList, NDimensions, TCompositeTransform, TOutputTransformPrecisionType>::PrintSelf(
  std::ostream & os,
  Indent         indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Input Transform: " << this->m_InputTransform << std::endl;
  os << indent << "Output Transform: " << this->m_Output << std::endl;
  os << indent << "Internal Transform Time: " << this->m_InternalTransformTime << std::endl;
  os << indent << "Explicit Mode: " << this->m_ExplicitMode << std::endl;
}


} // end namespace itk

#endif
