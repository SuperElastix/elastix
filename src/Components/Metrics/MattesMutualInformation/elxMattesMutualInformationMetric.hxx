#ifndef __elxMattesMutualInformationMetric_HXX__
#define __elxMattesMutualInformationMetric_HXX__

#include "elxMattesMutualInformationMetric.h"
#include "vnl/vnl_math.h"
#include <string>

namespace elastix
{
using namespace itk;


	/**
	 * ********************* Constructor ****************************
	 */

	template <class TElastix>
		MattesMutualInformationMetric<TElastix>
		::MattesMutualInformationMetric()
	{
		/** Initialize.*/
		this->m_ShowExactMetricValue = false;
		this->m_SamplesOnUniformGrid = false;

	} // end Constructor


	/**
	 * ************************ BeforeAll ***************************
	 */
	
	template <class TElastix>
		int MattesMutualInformationMetric<TElastix>
		::BeforeAll(void)
	{
		/** Return a value.*/
		return 0;

	} // end BeforeAll


	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void MattesMutualInformationMetric<TElastix>
		::Initialize(void) throw (ExceptionObject)
	{
		TimerPointer timer = TimerType::New();
		timer->StartTimer();
		this->Superclass1::Initialize();
		timer->StopTimer();
		elxout << "Initialization of MattesMutualInformation metric took: "
			<< static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

	} // end Initialize

	
	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void MattesMutualInformationMetric<TElastix>
		::BeforeEachResolution(void)
	{
		/** \todo Adapt SecondOrderRegularisationMetric.
		 * Set alpha, which balances the similarity and deformation energy
		 * E_total = (1-alpha)*E_sim + alpha*E_def.
		 * 	metric->SetAlpha( config.GetAlpha(level) );
		 */

		/** Get the current resolution level.*/
		unsigned int level = 
			( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();
		
		/** Set the number of histogram bins and spatial samples.*/				
		unsigned int numberOfHistogramBins = 32;
		unsigned int numberOfSpatialSamples = 10000;
		/** \todo guess the default numberOfSpatialSamples from the 
		 * imagesize, the numberOfParameters, and the number of bins...
		 */
		
		/** Read the parameters from the ParameterFile.*/
		this->m_Configuration->ReadParameter( numberOfHistogramBins, "NumberOfHistogramBins", level );
		this->m_Configuration->ReadParameter( numberOfSpatialSamples, "NumberOfSpatialSamples", level );
		
		/** Set them.*/
		this->SetNumberOfHistogramBins( numberOfHistogramBins );
		this->SetNumberOfSpatialSamples( numberOfSpatialSamples );

		/** Check if the exact metric value, computed on all pixels, should be shown, 
		 * and whether the all pixels should be used during optimisation
		 */

		/** Remove the ExactMetric-column, if it already existed. */
		xl::xout["iteration"].RemoveTargetCell("ExactMetric");

		bool useAllPixelsBool = false;
		std::string useAllPixels = "false";
		this->GetConfiguration()->
			ReadParameter(useAllPixels, "UseAllPixels", level);
		if (useAllPixels == "true")
		{
			useAllPixelsBool = true;
		}
		else
		{
			useAllPixelsBool = false;
		}
		this->SetUseAllPixels(useAllPixelsBool);

		if (!useAllPixelsBool)
		{
			/** Show the exact metric VALUE anyway? */
			std::string showExactMetricValue = "false";
			this->GetConfiguration()->
				ReadParameter(showExactMetricValue, "ShowExactMetricValue", level);
			if (showExactMetricValue == "true")
			{
				this->m_ShowExactMetricValue = true;
				xl::xout["iteration"].AddTargetCell("ExactMetric");
				xl::xout["iteration"]["ExactMetric"] << std::showpoint << std::fixed;
			}
			else
			{
				this->m_ShowExactMetricValue = false;
			}
		}
		else	
		{
			/** The exact metric value is shown anyway */
			this->m_ShowExactMetricValue = false;
		}
    
		/** Put spatial samples on a uniform grid? */
		this->m_SamplesOnUniformGrid = false;
		std::string samplesOnUniformGrid;
		this->GetConfiguration()->
			ReadParameter(samplesOnUniformGrid , "SamplesOnUniformGrid", level);
		if ( samplesOnUniformGrid == "true" )
		{
			this->m_SamplesOnUniformGrid = true;
	
			/** Read the desired spacing of the samples */
			unsigned int spacing_dim;
			for (unsigned int dim = 0; dim < FixedImageDimension; dim++)
			{
				spacing_dim = 2;
				this->GetConfiguration()->ReadParameter(
					spacing_dim,
					"SampleGridSpacing",
					level*FixedImageDimension + dim );
				this->m_SampleGridSpacing[dim] =
					static_cast<SampleGridSpacingValueType>( spacing_dim ); 
			}

		} // end if samplesOnUniformGrid
		
	} // end BeforeEachResolution
	


	/**
	 * ***************AfterEachIteration ****************************
	 */

	template <class TElastix>
		void MattesMutualInformationMetric<TElastix>
		::AfterEachIteration(void)
	{		
		/** Show the mutual information computed on all voxels,
		 * if the user wanted it */
		if (this->m_ShowExactMetricValue)
		{
			xl::xout["iteration"]["ExactMetric"] << this->GetExactValue(
				this->GetElastix()->
				GetElxOptimizerBase()->GetAsITKBaseType()->
				GetCurrentPosition() );
		}

	}


	/**
	 * *************** SelectNewSamples ****************************
	 */

	template <class TElastix>
		void MattesMutualInformationMetric<TElastix>
		::SelectNewSamples(void)
	{

		/** Select new spatial samples; only if we do not use ALL pixels
		 * anyway */
		if ( (!this->GetUseAllPixels())  && (!this->m_SamplesOnUniformGrid) )
		{
			/**
			* Allocate memory for the fixed image sample container.
			*/
			this->m_FixedImageSamples.resize( this->GetNumberOfSpatialSamples() );
	
			/** 
			* Uniformly sample the fixed image (within the fixed image region)
			* to create the sample points list.
			*/
			this->SampleFixedImageDomain( this->m_FixedImageSamples );
				
		} //end if 

	} // end SelectNewSamples

  
	/** 
	 * **************** SampleFixedImageDomain ***********************
	 *
	 * If desired, samples are chosen on a regular grid.
	 * This functionality is probably temporary, just for tests by Stefan.
	 */

	template <class TElastix>
		void MattesMutualInformationMetric<TElastix>
		::SampleFixedImageDomain( FixedImageSpatialSampleContainer& samples )
	{
		
		if ( !(this->m_SamplesOnUniformGrid) )
		{
			this->Superclass1::SampleFixedImageDomain(samples);
			return;
		}
    
		const FixedImageType * fi = this->GetFixedImage();
		SampleGridSizeType sampleGridSize;
		SampleGridIndexType sampleGridIndex;
		SampleGridIndexType index;
		const FixedImageSizeType & fixedImageSize =
			this->GetFixedImageRegion().GetSize();
		unsigned long numberOfSamplesOnGrid = 1;

		for (unsigned int dim = 0; dim < FixedImageDimension; dim++)
		{

			/** the number of sample point along one dimension */
			sampleGridSize[dim] = 1 + 
				(( fixedImageSize[dim] - 1 ) / this->m_SampleGridSpacing[dim]);

			/** the position of the first sample along this dimension is 
			 * chosen to center the grid nicely on the image */
			sampleGridIndex[dim] = (   fixedImageSize[dim] - 
				( (sampleGridSize[dim] - 1) * this->m_SampleGridSpacing[dim] +1 )   ) / 2;

			numberOfSamplesOnGrid *= sampleGridSize[dim];
			
		}
		
		samples.resize(numberOfSamplesOnGrid);			
		typename FixedImageSpatialSampleContainer::iterator iter = samples.begin();
		
		unsigned int dim_z = 1;
		if (FixedImageDimension > 2)
		{
			dim_z = sampleGridSize[2];
		}

		index = sampleGridIndex;

		if ( !(this->GetFixedImageMask()) )
		{

			//ugly loop
			for ( unsigned int z = 0; z < dim_z; z++)
			{
				for ( unsigned int y = 0; y < sampleGridSize[1]; y++)
				{
					for ( unsigned int x = 0; x < sampleGridSize[0]; x++)
					{
						// Get sampled fixed image value
						(*iter).FixedImageValue = fi->GetPixel(index);
						// Translate index to point
						fi->TransformIndexToPhysicalPoint(
							index, (*iter).FixedImagePointValue );
			 			// Jump to next position on grid
						index[0] += this->m_SampleGridSpacing[0];
						// Go to next position in sample container
						++iter;
					} // end x
					index[0] = sampleGridIndex[0];
					index[1] += this->m_SampleGridSpacing[1];
				} // end y
				if (FixedImageDimension > 2)
				{
					index[1] = sampleGridIndex[1];
				  index[2] += this->m_SampleGridSpacing[2];
				}
		  } // end z
			
		} // end if no mask
		else
		{
		  unsigned long nrOfValidSamples = 0;
			//ugly loop
			for ( unsigned int z = 0; z < dim_z; z++)
			{
				for ( unsigned int y = 0; y < sampleGridSize[1]; y++)
				{
					for ( unsigned int x = 0; x < sampleGridSize[0]; x++)
					{
						// Translate index to point
						this->m_FixedImage->TransformIndexToPhysicalPoint(
							index, (*iter).FixedImagePointValue );
			 			if (  this->m_FixedImageMask->IsInside( (*iter).FixedImagePointValue )  )
						{
							// Get sampled fixed image value
							(*iter).FixedImageValue = fi->GetPixel(index);
							// Go to next position in sample container
							++iter;
							++nrOfValidSamples;
						} // end if in mask
						// Jump to next position on grid
						index[0] += this->m_SampleGridSpacing[0];
					} // end x
					index[0] = sampleGridIndex[0];
					index[1] += this->m_SampleGridSpacing[1];
				} // end y
				if (FixedImageDimension > 2)
				{
					index[1] = sampleGridIndex[1];
					index[2] += this->m_SampleGridSpacing[2];
				}
		  } // end z

      samples.resize(nrOfValidSamples);   
					
		} // else (if mask exists)

		unsigned long nrOfSpatialSamples = samples.size();
		this->SetNumberOfSpatialSamples(nrOfSpatialSamples );
		
		/** Print some info */
		elxout << "\nMetric: " << nrOfSpatialSamples <<
			" spatial image samples have been selected on a uniform grid of "
			<< sampleGridSize[0] << "x" << sampleGridSize[1] << "x" << dim_z <<
			".\n" << std::endl;
		 
	} // end SampleFixedImageDomain.


} // end namespace elastix


#endif // end #ifndef __elxMattesMutualInformationMetric_HXX__

