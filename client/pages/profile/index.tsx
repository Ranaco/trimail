import * as React from "react";
import { Stack, Typography, Box, useTheme, Grid } from "@mui/material";
import { AppState } from "../_app";
import Capsule from "../../components/capsule";
import { usePolybase } from "@polybase/react";
import ProfileAvatar from "../../components/profile-avatar";
import ActiveButton from "../../components/active-button";

type FormStateProps = {
  firstName: string;
  lastName: string;
  email: string;
};

const Profile: React.FC = () => {
  const theme = useTheme();
  const { state, setState } = React.useContext(AppState);
  const polyDB = usePolybase();
  const filters: string[] = [
    "Data Science",
    "Innovation",
    "Future of Technology",
    "Blockchain",
    "VR",
    "Big Data",
    "Machine Learning",
    "IoT",
    "AR",
    "Fintech",
    "Healthcare",
    "User Experience",
    "Productivity",
  ];
  const frequency = [
    "Daily",
    "Weekly",
    "Monthly",
    "Monday | Wednesday | Friday",
    "Weekends",
  ];
  const [activeFilters, setActiveFilter] = React.useState<string[]>([]);
  const [file, setFile] = React.useState<any[] | undefined>();
  const [profile, setProfile] = React.useState<string | undefined>();
  const [userFrequency, setFrequency] = React.useState<string>();
  const [formState, setFormState] = React.useState<FormStateProps>({
    firstName: "",
    lastName: "",
    email: "",
  });

  const uploadProfile = () => {
    console.log(file);
  };
  const updateFrequency = (value: string) => {
    setFrequency(value);
    polyDB
      .collection("UserSBT")
      .record(String(state.user.id))
      .call("updateFrequency", [value]);
  };

  const updateFormState = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setFormState((val) => ({
      ...val,
      [e.target.name]: e.target.value,
    }));
  };

  const updateUserInterests = (interest: string, active: boolean) => {
    if (active) {
      setActiveFilter((val) => [...val, interest]);
      polyDB
        .collection("UserSBT")
        .record(String(state.user.id))
        .call("updateInterests", [[...activeFilters, interest]]);
    } else {
      setActiveFilter((val) => val.filter((k) => k === interest));
      polyDB
        .collection("UserSBT")
        .record(String(state.user.id))
        .call("updateInterests", [
          activeFilters.filter((val) => val !== interest),
        ]);
    }
  };

  React.useState(() => {
    if (state.user) {
      setActiveFilter(state.user.interests);
      setFrequency(state.user.frequency ?? frequency[0]);
    }
  });

  return state.user ? (
    <Stack direction={"row"} height="100%" gap={"20px"}>
      <Stack
        p={"20px"}
        bgcolor={theme.palette.secondary.dark}
        flex={1}
        borderRadius={"10px"}
        alignItems={"center"}
      >
        <ProfileAvatar
          file={file}
          profile={state.user.profileUrl}
          setFile={setFile}
          uploadProfile={uploadProfile}
        />
      </Stack>
      <Stack flex={1} gap={"20px"}>
        <Box
          bgcolor={theme.palette.secondary.dark}
          borderRadius={"10px"}
          flex={1}
          p={"20px"}
        >
          <Typography fontSize={"1.7em"}>
            Your chosen topics of interest
          </Typography>

          <Box mt={"40px"}>
            <Grid container spacing={3}>
              {filters.map((e, index) => {
                return (
                  <Grid key={index} item>
                    <Capsule
                      active={activeFilters.includes(e)}
                      label={e}
                      onChange={(label, checked) => {
                        console.log(label, checked);
                        updateUserInterests(label, checked);
                      }}
                    />
                  </Grid>
                );
              })}
            </Grid>
          </Box>
        </Box>
        <Box
          bgcolor={theme.palette.secondary.dark}
          borderRadius={"10px"}
          flex={0.5}
          p={"20px"}
        >
          <Typography fontSize={"1.7em"}>
            Choose the frequency of mails
          </Typography>
          <Grid mt={"30px"} container gap={"45px"}>
            {frequency.map((e, index) => (
              <Grid item key={index}>
                <ActiveButton
                  callback={updateFrequency}
                  currentActive={userFrequency}
                  title={e}
                />
              </Grid>
            ))}
          </Grid>
        </Box>
      </Stack>
    </Stack>
  ) : undefined;
};

export default Profile;
