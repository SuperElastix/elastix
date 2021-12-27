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
#ifndef itkVectorContainerSource_hxx
#define itkVectorContainerSource_hxx

#include "itkVectorContainerSource.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <class TOutputVectorContainer>
VectorContainerSource<TOutputVectorContainer>::VectorContainerSource()
{
  // Create the output. We use static_cast<> here because we know the default
  // output must be of type TOutputVectorContainer
  OutputVectorContainerPointer output = static_cast<TOutputVectorContainer *>(this->MakeOutput(0).GetPointer());

  this->ProcessObject::SetNumberOfRequiredOutputs(1);
  this->ProcessObject::SetNthOutput(0, output.GetPointer());

  this->m_GenerateDataRegion = 0;
  this->m_GenerateDataNumberOfRegions = 0;

} // end Constructor


/**
 * ******************* MakeOutput *******************
 */

template <class TOutputVectorContainer>
auto
VectorContainerSource<TOutputVectorContainer>::MakeOutput(unsigned int itkNotUsed(idx)) -> DataObjectPointer
{
  return static_cast<DataObject *>(TOutputVectorContainer::New().GetPointer());
} // end MakeOutput()


/**
 * ******************* GetOutput *******************
 */

template <class TOutputVectorContainer>
auto
VectorContainerSource<TOutputVectorContainer>::GetOutput() -> OutputVectorContainerType *
{
  if (this->GetNumberOfOutputs() < 1)
  {
    return 0;
  }

  return static_cast<OutputVectorContainerType *>(this->ProcessObject::GetOutput(0));
} // end GetOutput()

/**
 * ******************* GetOutput *******************
 */

template <class TOutputVectorContainer>
auto
VectorContainerSource<TOutputVectorContainer>::GetOutput(unsigned int idx) -> OutputVectorContainerType *
{
  return static_cast<OutputVectorContainerType *>(this->ProcessObject::GetOutput(idx));
} // end GetOutput()

/**
 * ******************* GraftOutput *******************
 */

template <class TOutputVectorContainer>
void
VectorContainerSource<TOutputVectorContainer>::GraftOutput(DataObject * graft)
{
  this->GraftNthOutput(0, graft);
} // end GraftOutput()


/**
 * ******************* GraftNthOutput *******************
 */

template <class TOutputVectorContainer>
void
VectorContainerSource<TOutputVectorContainer>::GraftNthOutput(unsigned int idx, DataObject * graft)
{
  /** Check idx. */
  if (idx >= this->GetNumberOfOutputs())
  {
    itkExceptionMacro(<< "Requested to graft output " << idx << " but this filter only has "
                      << this->GetNumberOfOutputs() << " Outputs.");
  }

  /** Check graft. */
  if (!graft)
  {
    itkExceptionMacro(<< "Requested to graft output that is a NULL pointer");
  }

  /** Get a pointer to the output. */
  DataObject * output = this->GetOutput(idx);

  /** Call Graft on the vector container in order to
   * copy meta-information, and containers. */
  output->Graft(graft);

} // end GraftNthOutput()


/**
 * ******************* PrintSelf *******************
 */

template <class TOutputVectorContainer>
void
VectorContainerSource<TOutputVectorContainer>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  //   int m_GenerateDataRegion;
  //   int m_GenerateDataNumberOfRegions;
} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef itkVectorContainerSource_hxx
