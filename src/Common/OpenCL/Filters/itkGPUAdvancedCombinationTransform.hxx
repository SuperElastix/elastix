/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUAdvancedCombinationTransform_hxx
#define __itkGPUAdvancedCombinationTransform_hxx

#include "itkGPUAdvancedCombinationTransform.h"
#include "itkGPUMatrixOffsetTransformBase.h"
#include <iomanip>

// begin of unnamed namespace
namespace
{
} // end of unnamed namespace

//------------------------------------------------------------------------------
namespace itk
{
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
GPUAdvancedCombinationTransform< TScalarType, NDimensions, TParentImageFilter >::GPUAdvancedCombinationTransform()
//:Superclass( NDimensions )
{
  // Add GPUMatrixOffsetTransformBase header
  const std::string sourcePath0(GPUMatrixOffsetTransformBaseHeaderKernel::GetOpenCLSource());
  m_Sources.push_back(sourcePath0);

  // Add GPUMatrixOffsetTransformBase source
  const std::string sourcePath1(GPUMatrixOffsetTransformBaseKernel::GetOpenCLSource());
  m_Sources.push_back(sourcePath1);

  m_SourcesLoaded = true; // we set it to true, sources are loaded from strings

  const unsigned int InputDimension  = InputSpaceDimension;
  const unsigned int OutputDimension = OutputSpaceDimension;

  this->m_ParametersDataManager->Initialize();
  this->m_ParametersDataManager->SetBufferFlag(CL_MEM_READ_ONLY);

  switch(OutputDimension)
  {
  case 1:
    this->m_ParametersDataManager->SetBufferSize(sizeof(GPUMatrixOffsetTransformBase1D));
    break;
  case 2:
    this->m_ParametersDataManager->SetBufferSize(sizeof(GPUMatrixOffsetTransformBase2D));
    break;
  case 3:
    this->m_ParametersDataManager->SetBufferSize(sizeof(GPUMatrixOffsetTransformBase3D));
    break;
  default: break;
  }

  this->m_ParametersDataManager->Allocate();
}

//------------------------------------------------------------------------------
template<class TScalarType, unsigned int NDimensions, class TParentImageFilter >
GPUDataManager::Pointer GPUAdvancedCombinationTransform<TScalarType, NDimensions, TParentImageFilter>
  ::GetParametersDataManager() const
{
  const unsigned int InputDimension  = InputSpaceDimension;
  const unsigned int OutputDimension = OutputSpaceDimension;
  const SpaceDimensionToType<InputSpaceDimension>  idim = {};
  const SpaceDimensionToType<OutputSpaceDimension> odim = {};

  switch(OutputDimension)
  {
  case 1:
    {
      GPUMatrixOffsetTransformBase1D transformBase;

      //SetMatrix1<ScalarType>(this->GetMatrix(), transformBase.Matrix, odim, idim);
      //SetOffset1<ScalarType>(this->GetOffset(), transformBase.Offset, odim);
      //SetMatrix1<ScalarType>(this->GetInverseMatrix(), transformBase.InverseMatrix, idim, odim);
      this->m_ParametersDataManager->SetCPUBufferPointer(&transformBase);
    }
    break;
  case 2:
    {
      GPUMatrixOffsetTransformBase2D transformBase;

      //SetMatrix2<ScalarType>(this->GetMatrix(), transformBase.Matrix, odim, idim);
      //SetOffset2<ScalarType>(this->GetOffset(), transformBase.Offset, odim);
      //SetMatrix2<ScalarType>(this->GetInverseMatrix(), transformBase.InverseMatrix, idim, odim);
      this->m_ParametersDataManager->SetCPUBufferPointer(&transformBase);
    }
    break;
  case 3:
    {
      GPUMatrixOffsetTransformBase3D transformBase;

      //SetMatrix3<ScalarType>(this->GetMatrix(), transformBase.Matrix, odim, idim);
      //SetOffset3<ScalarType>(this->GetOffset(), transformBase.Offset, odim);
      //SetMatrix3<ScalarType>(this->GetInverseMatrix(), transformBase.InverseMatrix, idim, odim);
      this->m_ParametersDataManager->SetCPUBufferPointer(&transformBase);
    }
    break;
  default: break;
  }

  this->m_ParametersDataManager->SetGPUDirtyFlag(true);
  this->m_ParametersDataManager->UpdateGPUBuffer();

  return this->m_ParametersDataManager;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUAdvancedCombinationTransform< TScalarType, NDimensions, TParentImageFilter >
  ::GetSourceCode(std::string &_source) const
{
  if(!m_SourcesLoaded)
    return false;

  // Create the final source code
  std::ostringstream source;
  // Add other sources
  for(unsigned int i=0; i<m_Sources.size(); i++)
  {
    source << m_Sources[i] << std::endl;
  }
  _source = source.str();
  return true;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
void GPUAdvancedCombinationTransform< TScalarType, NDimensions, TParentImageFilter >
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

} // namespace

#endif
